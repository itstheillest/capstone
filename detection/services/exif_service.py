"""
EXIF metadata extraction and discrepancy detection.
"""

import logging
import shutil
from datetime import datetime, timezone as dt_timezone
from PIL.TiffImagePlugin import IFDRational

logger = logging.getLogger(__name__)

KNOWN_EDITOR_TAGS = [
    "photoshop",
    "gimp",
    "lightroom",
    "affinity photo",
    "paint.net",
    "canva",
    "snapseed",
    "faceapp",
    "picsart",
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dalle",
    "firefly",
    "comfyui",
    "automatic1111",
    "novelai",
    "imagen",
    "flux",
    "gemini",
    "c2pa",
    "bing image creator",
]

_INFO_METADATA_KEYS = (
    "Software",
    "software",
    "parameters",
    "prompt",
    "workflow",
    "Generator",
    "Comment",
    "Description",
)


def _sanitize_value(value):
    """
    Recursively converts EXIF metadata values (such as IFDRational, bytes, or tuples)
    into standard JSON-serializable Python primitives.
    """
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, IFDRational):
        return float(value) if value.denominator != 0 else 0.0
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _parse_exif_datetime(value):
    """Parses EXIF timestamps into timezone-aware UTC datetimes for Django DateTimeField."""
    if not value:
        return None
    cleaned = str(value).strip()[:19]
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=dt_timezone.utc)
        except ValueError:
            continue
    return None


def _rational_to_float(val) -> float | None:
    if isinstance(val, IFDRational):
        return float(val) if val.denominator != 0 else None
    if isinstance(val, (tuple, list)) and len(val) == 2:
        num, den = val
        return float(num) / float(den) if den else None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _dms_to_decimal(dms, ref) -> float | None:
    """Converts EXIF DMS (degrees, minutes, seconds) plus hemisphere ref into decimal degrees."""
    if isinstance(dms, (int, float)):
        decimal = float(dms)
    elif isinstance(dms, (tuple, list)) and len(dms) == 3:
        parts = [_rational_to_float(p) for p in dms]
        if any(p is None for p in parts):
            return None
        deg, mins, secs = parts
        decimal = deg + (mins / 60.0) + (secs / 3600.0)
    else:
        return None

    ref_str = str(ref or "").strip().upper()
    if ref_str in ("S", "W"):
        decimal = -abs(decimal)
    return round(decimal, 6)


def _analyze_discrepancies(raw_exif: dict, software_tag: str):
    notes = []
    has_discrepancy = False

    if not raw_exif or len(raw_exif) <= 2:
        has_discrepancy = True
        notes.append("No EXIF metadata present — may indicate stripping, re-saving, or synthetic origin.")

    if software_tag:
        lowered = software_tag.lower()
        for editor in KNOWN_EDITOR_TAGS:
            if editor in lowered:
                has_discrepancy = True
                notes.append(f"Software tag indicates editing/generative tool: '{software_tag}'.")
                break

    return has_discrepancy, "; ".join(notes)


def run_exiftool(image_path: str) -> dict:
    import exiftool

    with exiftool.ExifToolHelper() as et:
        results = et.get_metadata(image_path)
    return results[0] if results else {}


def run_pillow_fallback(image_path: str) -> dict:
    """Extracts IFD0, Exif sub-IFD, GPSInfo sub-IFD, and PNG/container info tags via Pillow."""
    from PIL import ExifTags, Image

    try:
        with Image.open(image_path) as img:
            raw = img.getexif()
            result = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw.items()} if raw else {}

            if raw:
                # Extract Exif sub-IFD (DateTimeOriginal, lens metadata, etc.)
                exif_ifd_tag = getattr(ExifTags.IFD, "Exif", 0x8769) if hasattr(ExifTags, "IFD") else 0x8769
                try:
                    exif_ifd = raw.get_ifd(exif_ifd_tag) or {}
                    for k, v in exif_ifd.items():
                        key_name = ExifTags.TAGS.get(k, str(k))
                        if key_name not in result:
                            result[key_name] = v
                except Exception:
                    pass

                # Extract GPSInfo sub-IFD and convert DMS to signed decimal coordinates
                gps_ifd_tag = getattr(ExifTags.IFD, "GPSInfo", 0x8825) if hasattr(ExifTags, "IFD") else 0x8825
                try:
                    gps_ifd = raw.get_ifd(gps_ifd_tag) or {}
                    if gps_ifd:
                        gps_named = {ExifTags.GPSTAGS.get(k, str(k)): v for k, v in gps_ifd.items()}
                        lat = _dms_to_decimal(gps_named.get("GPSLatitude"), gps_named.get("GPSLatitudeRef"))
                        lon = _dms_to_decimal(gps_named.get("GPSLongitude"), gps_named.get("GPSLongitudeRef"))
                        if lat is not None:
                            result["GPSLatitude"] = lat
                        if lon is not None:
                            result["GPSLongitude"] = lon
                except Exception:
                    pass

            # Inspect container text/PNG chunks for editing or generative AI software tags
            info = img.info or {}
            for key in _INFO_METADATA_KEYS:
                val = info.get(key)
                if val:
                    text_val = str(val).strip()
                    if text_val and "Software" not in result:
                        result["Software"] = text_val[:150]
                    if text_val and f"Info:{key}" not in result:
                        result[f"Info:{key}"] = text_val[:500]

            return result
    except Exception as exc:
        logger.warning("Pillow EXIF extraction failed: %s", exc)
        return {}


def extract_exif(image_path: str) -> dict:
    """
    Returns a dict with normalized fields ready to populate ExifMetadata.
    Safely falls back to Pillow if exiftool binary or PyExifTool package is unavailable.
    """
    used_exiftool = False
    raw_exif = {}

    if shutil.which("exiftool") is not None:
        try:
            raw_exif = run_exiftool(image_path)
            used_exiftool = True
        except Exception as exc:
            logger.warning("ExifTool execution failed (%s); falling back to Pillow.", exc)
            raw_exif = run_pillow_fallback(image_path)
    else:
        raw_exif = run_pillow_fallback(image_path)

    raw_exif = _sanitize_value(raw_exif)

    camera_make = raw_exif.get("EXIF:Make") or raw_exif.get("Make", "") or ""
    camera_model = raw_exif.get("EXIF:Model") or raw_exif.get("Model", "") or ""
    software_tag = (
        raw_exif.get("EXIF:Software")
        or raw_exif.get("PNG:Software")
        or raw_exif.get("Software", "")
        or ""
    )
    capture_raw = (
        raw_exif.get("EXIF:DateTimeOriginal")
        or raw_exif.get("DateTimeOriginal")
        or raw_exif.get("EXIF:CreateDate")
        or raw_exif.get("DateTimeDigitized")
        or raw_exif.get("DateTime", "")
    )
    gps_lat = raw_exif.get("EXIF:GPSLatitude") if "EXIF:GPSLatitude" in raw_exif else raw_exif.get("GPSLatitude")
    gps_lon = raw_exif.get("EXIF:GPSLongitude") if "EXIF:GPSLongitude" in raw_exif else raw_exif.get("GPSLongitude")

    has_discrepancy, notes = _analyze_discrepancies(raw_exif, str(software_tag))

    return {
        "raw_exif": raw_exif,
        "camera_make": str(camera_make).strip()[:100],
        "camera_model": str(camera_model).strip()[:100],
        "software_tag": str(software_tag).strip()[:150],
        "capture_timestamp": _parse_exif_datetime(capture_raw),
        "gps_latitude": float(gps_lat) if isinstance(gps_lat, (int, float)) else None,
        "gps_longitude": float(gps_lon) if isinstance(gps_lon, (int, float)) else None,
        "has_discrepancy": has_discrepancy,
        "discrepancy_notes": notes,
        "used_exiftool": used_exiftool,
    }
