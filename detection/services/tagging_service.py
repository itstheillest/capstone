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


def generate_tags(face_count: int, verdict: str, manipulation_score: float) -> list[dict]:
    """
    Returns a list of dicts ready to populate TaggingResult:
    tag_code, tag_label, rule_description, confidence.
    """
    tags = []
    for rule in RULES:
        result = rule(face_count, verdict, manipulation_score)
        if result:
            tags.append(result)
    return tags