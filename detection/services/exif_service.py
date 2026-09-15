"""
EXIF metadata extraction and discrepancy detection.

Prefers ExifTool (reads far more tags than Pillow, including maker notes and
XMP data editing tools leave behind) and falls back to Pillow's built-in EXIF
reader if the `exiftool` binary isn't installed on the host — so the pipeline
still runs (with reduced tag coverage) on a machine that skipped that setup
step.
"""

import shutil
from datetime import datetime
from PIL.TiffImagePlugin import IFDRational

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
]


def _sanitize_value(value):
    """
    Recursively converts EXIF metadata values (such as IFDRational, bytes, or tuples)
    into standard JSON-serializable Python primitives.
    """
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_sanitize_value(v) for v in value]
    elif isinstance(value, IFDRational):
        return float(value) if value.denominator != 0 else 0.0
    elif isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    elif isinstance(value, (int, float, str, bool)) or value is None:
        return value
    else:
        return str(value)


def _parse_exif_datetime(value):
    if not value:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


def _analyze_discrepancies(raw_exif: dict, software_tag: str):
    notes = []
    has_discrepancy = False

    if not raw_exif or len(raw_exif) <= 2:
        # exiftool always returns SourceFile/ExifToolVersion even for images
        # with genuinely no embedded metadata — anything at or below that is
        # effectively "no EXIF at all", which is itself a signal (common
        # after re-saving, screenshotting, or generation by an AI tool).
        has_discrepancy = True
        notes.append("No EXIF metadata present — may indicate stripping, re-saving, or synthetic origin.")

    if software_tag:
        lowered = software_tag.lower()
        for editor in KNOWN_EDITOR_TAGS:
            if editor in lowered:
                has_discrepancy = True
                notes.append(f"Software tag indicates editing tool: '{software_tag}'.")
                break

    return has_discrepancy, "; ".join(notes)


def run_exiftool(image_path: str) -> dict:
    import exiftool

    with exiftool.ExifToolHelper() as et:
        results = et.get_metadata(image_path)
    return results[0] if results else {}


def run_pillow_fallback(image_path: str) -> dict:
    from PIL import ExifTags, Image

    try:
        with Image.open(image_path) as img:
            raw = img.getexif()
            if not raw:
                return {}
            return {ExifTags.TAGS.get(k, str(k)): v for k, v in raw.items()}
    except Exception:
        return {}


def extract_exif(image_path: str) -> dict:
    """
    Returns a dict with normalized fields ready to populate ExifMetadata:
    raw_exif, camera_make, camera_model, software_tag, capture_timestamp,
    gps_latitude, gps_longitude, has_discrepancy, discrepancy_notes.
    """
    used_exiftool = shutil.which("exiftool") is not None

    if used_exiftool:
        raw_exif = run_exiftool(image_path)
    else:
        raw_exif = run_pillow_fallback(image_path)

    # Sanitize dictionary values to ensure full JSON serializability
    raw_exif = _sanitize_value(raw_exif)

    camera_make = raw_exif.get("EXIF:Make") or raw_exif.get("Make", "") or ""
    camera_model = raw_exif.get("EXIF:Model") or raw_exif.get("Model", "") or ""
    software_tag = raw_exif.get("EXIF:Software") or raw_exif.get("Software", "") or ""
    capture_raw = raw_exif.get("EXIF:DateTimeOriginal") or raw_exif.get("DateTimeOriginal", "")
    gps_lat = raw_exif.get("EXIF:GPSLatitude") or raw_exif.get("GPSLatitude")
    gps_lon = raw_exif.get("EXIF:GPSLongitude") or raw_exif.get("GPSLongitude")

    has_discrepancy, notes = _analyze_discrepancies(raw_exif, str(software_tag))

    return {
        "raw_exif": raw_exif,
        "camera_make": str(camera_make)[:100],
        "camera_model": str(camera_model)[:100],
        "software_tag": str(software_tag)[:150],
        "capture_timestamp": _parse_exif_datetime(capture_raw),
        "gps_latitude": float(gps_lat) if isinstance(gps_lat, (int, float)) else None,
        "gps_longitude": float(gps_lon) if isinstance(gps_lon, (int, float)) else None,
        "has_discrepancy": has_discrepancy,
        "discrepancy_notes": notes,
        "used_exiftool": used_exiftool,
    }