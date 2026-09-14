"""
Tagging Layer.

Combines what the YOLOv8 face detector found (objects of interest) with the
Synthetic Media Analysis Layer's verdict (from forensic_service) into a
semantic tag — e.g. "Face detected + Deepfake verdict = identity_fraud".

Each rule is a plain Python function so the logic is easy to read, test,
and extend; add new rules to RULES below rather than editing generate_tags()
directly. tag_code is the lookup key the Law Mapping Layer (legalmap app)
uses to find relevant Philippine legal provisions — keep codes stable once
you start seeding LegalMapping rows against them.
"""
from ultralytics import YOLO
from PIL import Image

# Model loads ONCE when worker process initializes
_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model

def generate_tags(image_path: str, face_count: int = 0, verdict: str = "", manipulation_score: float = 0.0) -> list[dict]:
    yolo = get_yolo_model()
    results = yolo(image_path, conf=0.4)
    tags = []
    seen = set()

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id]
            confidence = float(box.conf[0])

            if label not in seen:
                seen.add(label)
                tags.append({
                    "tag_code": label.lower().replace(" ", "_"),
                    "tag_label": label.capitalize(),
                    "confidence": round(confidence, 4),
                })

    return tags

def _rule_identity_fraud(face_count: int, verdict: str, manipulation_score: float):
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
            "tag_label": "Likely Authentic",
            "rule_description": (
                f"Forensic verdict was 'authentic' (score {manipulation_score:.3f}) "
                f"with {face_count} face(s) detected. No manipulation indicators found."
            ),
            "confidence": 1 - manipulation_score,
        }
    return None


RULES = [
    _rule_identity_fraud,
    _rule_manipulated_no_face,
    _rule_inconclusive,
    _rule_authentic,
]

