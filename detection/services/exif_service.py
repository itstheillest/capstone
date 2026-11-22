"""
EXIF metadata extraction and discrepancy detection.
"""

import logging
import shutil
from datetime import datetime
from PIL.TiffImagePlugin import IFDRational
import json
import os
from pathlib import Path

from celery import result

from c2pa import Context, Reader, Settings

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
]

AI_PROVENANCE_KEYWORDS = [
    "google ai",
    "made with google",
    "trainedalgorithmicmedia",
    "chatgpt",
    "dall-e",
    "dalle",
    "openai",
    "midjourney",
    "stable diffusion",
    "stablediffusion",
    "comfyui",
    "automatic1111",
    "adobe firefly",
    "flux",
    "ideogram",
    "recraft",
]

AI_SOURCE_TYPE = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/"
    "trainedAlgorithmicMedia"
)


def inspect_c2pa_ai_provenance(image_path: str) -> dict:
    """Inspect C2PA data; only trusted AI-generation assertions count as verified."""
    try:
        anchors_path = os.environ.get("C2PA_TRUST_ANCHORS_PATH")

        if anchors_path:
            anchors_file = Path(anchors_path)
            if not anchors_file.is_file():
                raise FileNotFoundError(
                    f"C2PA trust anchors not found: {anchors_path}"
                )

            anchors_pem = anchors_file.read_text(encoding="utf-8")
            settings = Settings.from_dict({
                "verify": {"verify_cert_anchors": True},
                "trust": {"trust_anchors": anchors_pem},
            })
            context = Context(settings)
        else:
            context = Context()

        with context as ctx:
            reader = Reader.try_create(str(image_path), context=ctx)

            if reader is None:
                return {
                    "status": "none",
                    "validation_state": None,
                    "reason": "No C2PA manifest is present.",
                }

            with reader:
                validation_state = reader.get_validation_state()
                manifest = reader.get_active_manifest()

        if not manifest:
            return {
                "status": "none",
                "validation_state": validation_state,
                "reason": "No active C2PA manifest was found.",
            }

        has_ai_source_type = any(
            action.get("digitalSourceType") == AI_SOURCE_TYPE
            for assertion in manifest.get("assertions", [])
            if assertion.get("label") in {"c2pa.actions", "c2pa.actions.v2"}
            for action in assertion.get("data", {}).get("actions", [])
        )

        if not has_ai_source_type:
            return {
                "status": "none",
                "validation_state": validation_state,
                "reason": "No AI-generation source type was found.",
            }

        if validation_state == "Trusted":
            return {
                "status": "verified_ai",
                "validation_state": validation_state,
                "reason": "Trusted C2PA manifest declares AI-generated content.",
            }

        return {
            "status": "unverified_ai_marker",
            "validation_state": validation_state,
            "reason": "AI-generation assertion found, but its signer is not trusted.",
        }

    except Exception as exc:
        logger.warning(
            "C2PA validation could not be completed for %s: %s",
            image_path,
            exc,
        )
        
        return {
            "status": "validation_error",
            "validation_state": None,
            "reason": str(exc),
        }


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
            return datetime.strptime(str(value)[:19], fmt)
        except ValueError:
            continue
    return None


def _find_ai_provenance_markers(raw_exif: dict) -> list[str]:
    """Scans all extracted EXIF/XMP/IPTC/PNG-text fields for AI generator markers."""
    markers = []
    for key, val in raw_exif.items():
        if key == "SourceFile":
            continue
        haystack = f"{key}: {val}".lower()
        for kw in AI_PROVENANCE_KEYWORDS:
            if kw in haystack and kw not in markers:
                markers.append(kw)
    return markers


def _analyze_discrepancies(raw_exif: dict, has_camera_info: bool, software_tag: str):
    notes = []
    has_discrepancy = False

    ai_markers = _find_ai_provenance_markers(raw_exif)
    ai_metadata_detected = len(ai_markers) > 0
    
    if ai_metadata_detected:
        has_discrepancy = True
        notes.append(f"AI provenance metadata detected: {', '.join(ai_markers)}.")

    if not has_camera_info:
        has_discrepancy = True
        notes.append(
            "No camera make, model, or capture time in the metadata. Common after re-sharing "
            "(e.g. via social media), screenshots, or re-saving; also seen in synthetic images. "
            "Not proof of manipulation."
        )

    if software_tag:
        lowered = software_tag.lower()
        for editor in KNOWN_EDITOR_TAGS:
            if editor in lowered:
                has_discrepancy = True
                notes.append(f"Software tag indicates editing/generative tool: '{software_tag}'.")
                break

    return has_discrepancy, "; ".join(notes), ai_metadata_detected, ai_markers


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
            info_software = img.info.get("Software") or img.info.get("parameters") or ""
            result = {ExifTags.TAGS.get(k, str(k)): v for k, v in raw.items()} if raw else {}

            # Preserve PNG text chunks / XMP / comment fields for AI provenance scanning
            for info_key, info_val in img.info.items():
                if info_key not in ("icc_profile", "exif") and info_key not in result:
                    result[str(info_key)] = _sanitize_value(info_val)

            info_software = img.info.get("Software") or img.info.get("parameters") or ""
            if info_software and "Software" not in result:
                result["Software"] = str(info_software)[:150]
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
    capture_raw = raw_exif.get("EXIF:DateTimeOriginal") or raw_exif.get("DateTimeOriginal", "")
    gps_lat = raw_exif.get("EXIF:GPSLatitude") or raw_exif.get("GPSLatitude")
    gps_lon = raw_exif.get("EXIF:GPSLongitude") or raw_exif.get("GPSLongitude")

    has_camera_info = bool(camera_make or camera_model or capture_raw)
    has_discrepancy, notes, ai_metadata_detected, ai_markers = _analyze_discrepancies(
        raw_exif, has_camera_info, str(software_tag)
    )
    
    c2pa_result = inspect_c2pa_ai_provenance(image_path)
    
    raw_exif["PixifAI:C2PA"] = {
        "status": c2pa_result["status"],
        "validation_state": c2pa_result["validation_state"],
        "reason": c2pa_result["reason"],
    }
    
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
        "ai_metadata_detected": ai_metadata_detected,
        "ai_metadata_markers": ai_markers,
        "c2pa_status": c2pa_result["status"],
        "c2pa_validation_state": c2pa_result["validation_state"],
        "c2pa_reason": c2pa_result["reason"],
    }