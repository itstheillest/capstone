"""
Tagging Layer.

Combines YOLOv8 object detections, EXIF/provenance metadata checks, and the
Forensic Service verdict into semantic and legal tags that drive the
Philippine Law Mapping Layer (legalmap app).
"""

import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _has_yolo = True
except ImportError:
    YOLO = None
    _has_yolo = False

_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if not _has_yolo:
        return None
    if _yolo_model is None:
        base_dir = Path(settings.BASE_DIR)
        candidates = [
            base_dir / "model_weights" / "yolov8n.pt",
            base_dir / "yolov8n.pt",
        ]
        model_ref = next((str(p) for p in candidates if p.exists()), str(base_dir / "yolov8n.pt"))
        try:
            _yolo_model = YOLO(model_ref)
        except Exception as exc:
            logger.warning("Could not load YOLOv8n model (%s); using rule-based tagging.", exc)
            return None
    return _yolo_model


def _rule_ai_metadata( provenance_status: str = "none", ai_metadata_markers: list[str] | None = None, **kwargs,):
    markers = ", ".join(ai_metadata_markers or [])

    if provenance_status == "verified_ai":
        return {
            "tag_code": "synthetic_media",
            "tag_label": "AI-Generated Media — Verified Provenance",
            "rule_description": (
                "A trusted C2PA manifest declares an AI-generation source type."
            ),
            # This describes verified provenance, not the visual classifier's score.
            "confidence": 1.0,
        }

    if provenance_status == "unverified_ai_marker":
        marker_text = f" Marker clues: {markers}." if markers else ""
        return {
            "tag_code": "requires_human_review",
            "tag_label": "Unverified AI Provenance — Requires Human Review",
            "rule_description": (
                "AI-related metadata was found, but its provenance could not be verified."
                f"{marker_text}"
            ),
            "confidence": None,
        }

    return None

def _rule_identity_fraud(face_count: int, verdict: str, manipulation_score: float, **kwargs):
    if face_count > 0 and verdict == "manipulated":
        return {
            "tag_code": "identity_fraud",
            "tag_label": "Possible Identity Fraud (Face + Manipulation Detected)",
            "rule_description": (
                f"{face_count} face(s) detected and forensic verdict was 'manipulated' "
                f"(score {manipulation_score:.3f}). Combination suggests a possible face "
                "swap, synthetic identity, or manipulated likeness."
            ),
            "confidence": manipulation_score,
        }
    return None


def _rule_manipulated_no_face(face_count: int, verdict: str, manipulation_score: float, **kwargs):
    if face_count == 0 and verdict == "manipulated":
        return {
            "tag_code": "manipulated_media",
            "tag_label": "Possible Manipulated Media (No Face Detected)",
            "rule_description": (
                f"No faces detected, but forensic verdict was 'manipulated' "
                f"(score {manipulation_score:.3f}). Likely non-portrait image splicing "
                "or synthetic generation."
            ),
            "confidence": manipulation_score,
        }
    return None


def _rule_inconclusive(face_count: int, verdict: str, manipulation_score: float, **kwargs):
    if verdict == "inconclusive":
        return {
            "tag_code": "requires_human_review",
            "tag_label": "Inconclusive — Requires Human Review",
            "rule_description": (
                f"Automated forensic indicators produced a score of {manipulation_score:.3f} "
                f"with {face_count} face(s) detected. This is not conclusive evidence of "
                "manipulation; manual review is required."
            ),
            "confidence": manipulation_score,
        }
    return None


def _rule_missing_provenance(face_count: int, verdict: str, manipulation_score: float, has_exif_discrepancy: bool = False, ai_metadata_detected: bool = False, ai_metadata_markers: list[str] | None = None, provenance_status: str = "none",): 
    if (provenance_status == "none" and not ai_metadata_detected and verdict == "authentic" and has_exif_discrepancy):
        return {
            "tag_code": "requires_human_review",
            "tag_label": "Unverified Provenance — Requires Human Review",
            "rule_description": (
                f"Visual classifier scored 'authentic' ({manipulation_score:.3f}), but "
                "camera EXIF metadata is missing or shows editing software. Routed to "
                "human review rather than auto-approving."
            ),
            "confidence": round(1.0 - manipulation_score, 4),
        }
    
    return None


def _rule_authentic(face_count: int, verdict: str, manipulation_score: float, has_exif_discrepancy: bool = False, ai_metadata_detected: bool = False, ai_metadata_markers: list[str] | None = None, provenance_status: str = "none",):
    if ( provenance_status == "none" and not ai_metadata_detected and verdict == "authentic" and not has_exif_discrepancy):
        # Keep the existing tag dictionary here.
        return {
            "tag_code": "likely_authentic",
            "tag_label": "Likely Authentic Media",
            "rule_description": (
                f"Forensic verdict was 'authentic' (score {manipulation_score:.3f}) "
                f"with valid camera metadata and {face_count} face(s) detected."
            ),
            "confidence": round(1.0 - manipulation_score, 4),
        }
    return None


RULES = [
    _rule_ai_metadata,
    _rule_manipulated_no_face,
    _rule_inconclusive,
    _rule_missing_provenance,
    _rule_authentic,
]


def generate_tags(
    image_path: str,
    face_count: int = 0,
    verdict: str = "",
    manipulation_score: float = 0.0,
    has_exif_discrepancy: bool = False,
    ai_metadata_detected: bool = False,
    ai_metadata_markers: list[str] | None = None,
    provenance_status: str = "none",
) -> list[dict]:
    
    tags = []
    seen = set()

    yolo = get_yolo_model()
    if yolo is not None:
        try:
            results = yolo(image_path, conf=0.4, verbose=False)
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    label = result.names[class_id]
                    confidence = float(box.conf[0])
                    code = label.lower().replace(" ", "_")

                    if code not in seen:
                        seen.add(code)
                        tags.append({
                            "tag_code": code,
                            "tag_label": label.capitalize(),
                            "confidence": round(confidence, 4),
                            "rule_description": f"YOLO object detection: '{label}' detected with confidence {confidence:.4f}.",
                        })
        except Exception as exc:
            logger.warning("YOLO object tagging failed (%s).", exc)

    elif face_count > 0:
        seen.add("person")
        tags.append({
            "tag_code": "person",
            "tag_label": "Person",
            "confidence": 0.95,
            "rule_description": f"Biometric detector identified {face_count} human face(s).",
        })

    # Always evaluate legal mapping rules regardless of whether YOLO is installed
    for rule_fn in RULES:
        result = rule_fn(
            face_count=face_count,
            verdict=verdict,
            manipulation_score=manipulation_score,
            has_exif_discrepancy=has_exif_discrepancy,
            ai_metadata_detected=ai_metadata_detected,
            ai_metadata_markers=ai_metadata_markers,
            provenance_status=provenance_status,
        )
        if result and result["tag_code"] not in seen:
            seen.add(result["tag_code"])
            tags.append(result)

    return tags
