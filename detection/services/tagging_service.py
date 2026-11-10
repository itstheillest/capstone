"""
Tagging Layer.

Combines YOLOv8 detections with the Forensic Service verdict into semantic
and legal tags that drive the Philippine Law Mapping Layer (legalmap app).
"""

import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _has_yolo = True
except ImportError:
    _has_yolo = False

_yolo_model = None


def get_yolo_model():
    global _yolo_model
    if not _has_yolo:
        return None
    if _yolo_model is None:
        try:
            _yolo_model = YOLO("yolov8n.pt")
        except Exception as exc:
            logger.warning("Could not load YOLOv8n model (%s); using rule-based tagging.", exc)
            return None
    return _yolo_model


def _rule_identity_fraud(face_count: int, verdict: str, manipulation_score: float):
    if face_count > 0 and verdict == "manipulated":
        return {
            "tag_code": "identity_fraud",
            "tag_label": "Identity Fraud / Profile Misrepresentation",
            "rule_description": (
                f"{face_count} face(s) detected and forensic verdict was 'manipulated' "
                f"(score {manipulation_score:.3f}). Combination suggests a possible face "
                "swap, synthetic identity, or manipulated likeness."
            ),
            "confidence": manipulation_score,
        }
    return None


def _rule_manipulated_no_face(face_count: int, verdict: str, manipulation_score: float):
    if face_count == 0 and verdict == "manipulated":
        return {
            "tag_code": "manipulated_media",
            "tag_label": "Manipulated Media (No Face Detected)",
            "rule_description": (
                f"No faces detected, but forensic verdict was 'manipulated' "
                f"(score {manipulation_score:.3f}). Likely non-portrait image splicing "
                "or synthetic generation."
            ),
            "confidence": manipulation_score,
        }
    return None


def _rule_inconclusive(face_count: int, verdict: str, manipulation_score: float):
    if verdict == "inconclusive":
        return {
            "tag_code": "requires_human_review",
            "tag_label": "Inconclusive — Requires Human Review",
            "rule_description": (
                f"Forensic score ({manipulation_score:.3f}) fell in the inconclusive "
                f"range with {face_count} face(s) detected. Flagged for manual review "
                "rather than an automated verdict."
            ),
            "confidence": manipulation_score,
        }
    return None


def _rule_authentic(face_count: int, verdict: str, manipulation_score: float):
    if verdict == "authentic":
        return {
            "tag_code": "likely_authentic",
            "tag_label": "Likely Authentic Media",
            "rule_description": (
                f"Forensic verdict was 'authentic' (score {manipulation_score:.3f}) "
                f"with {face_count} face(s) detected. No manipulation indicators found."
            ),
            "confidence": round(1.0 - manipulation_score, 4),
        }
    return None


RULES = [
    _rule_identity_fraud,
    _rule_manipulated_no_face,
    _rule_inconclusive,
    _rule_authentic,
]


def generate_tags(
    image_path: str,
    face_count: int = 0,
    verdict: str = "",
    manipulation_score: float = 0.0,
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
        result = rule_fn(face_count, verdict, manipulation_score)
        if result and result["tag_code"] not in seen:
            seen.add(result["tag_code"])
            tags.append(result)

    return tags