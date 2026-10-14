"""
Law Mapping Layer.

Looks up which LegalProvision rows a given tag_code (from the Tagging
Layer) implicates, and creates LegalMapping rows linking them to a
DetectionReport. This is deliberately a static lookup table, not something
inferred at runtime by a model — so a reviewer (or your panel) can audit
exactly which rule produced which legal citation, and so a lawyer can
review/edit TAG_TO_PROVISIONS without touching detection code.

Update TAG_TO_PROVISIONS as your Tagging Layer's rule set grows in
detection/services/tagging_service.py — tag_code values must match exactly.

Phase 4 additions:
- Covers all seeded LegalProvision rows (privacy, photo-voyeurism, fact-check debunked).
- apply_mappings() now accepts optional enrichment kwargs so the pipeline can
  pass in face_count, reverse_match_count, and debunked_rating_found after the
  fact-check layer runs, triggering additional provisions without changing the
  core tag-code contract.
- generate_report_summary() aggregates all LegalMapping rows for a report into
  a structured dict ready for the API serializer.
"""

import logging

from legalmap.models import LegalMapping, LegalProvision

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TAG → PROVISION lookup table
# ---------------------------------------------------------------------------
# Keys   : tag_code strings emitted by detection/services/tagging_service.py
# Values : list of (law_number, section) tuples matching LegalProvision rows
#          seeded by seed_legal_provisions.py
#
# Sentinel tag codes used internally by apply_mappings() for enrichment-path
# triggers (not emitted by the Tagging Layer directly):
#   "face_privacy_violation"  – face detected + no consent signal
#   "debunked_fact_check"     – fact-check API returned a FALSE/MISLEADING rating
#   "reverse_image_hit"       – reverse image search returned ≥1 match
# ---------------------------------------------------------------------------

TAG_TO_PROVISIONS: dict[str, list[tuple[str, str]]] = {
    # --- Tagging Layer rule-based codes ---
    "identity_fraud": [
        ("RA 10175", "Sec. 4(b)(3)"),   # Computer-related Identity Theft
        ("Act No. 3815", "Art. 315"),   # Estafa
        ("HB 3214", ""),                # Deepfake Regulation Act (pending)
        ("SB 758", ""),                 # Deepfake Regulation and Digital Identity Protection Act (pending)
    ],
    "manipulated_media": [
        ("RA 10175", "Sec. 4(c)(4)"),   # Cyberlibel
        ("HB 3214", ""),                # Deepfake Regulation Act (pending)
        ("SB 758", ""),                 # Deepfake Regulation and Digital Identity Protection Act (pending)
    ],
    "requires_human_review": [
        # Inconclusive verdict — flag cyberlibel and pending deepfake bills for manual triage
        ("RA 10175", "Sec. 4(c)(4)"),
        ("HB 3214", ""),
    ],

    # --- Enrichment-path sentinel codes (set by apply_mappings kwargs) ---
    "face_privacy_violation": [
        ("RA 10173", "Sec. 25"),        # Data Privacy Act — unauthorized processing of biometrics
        ("RA 9995", "Sec. 4"),          # Anti-Photo and Video Voyeurism Act
        ("SB 758", ""),                 # Digital Identity Protection Act (pending)
    ],
    "debunked_fact_check": [
        ("RA 10175", "Sec. 4(c)(4)"),   # Cyberlibel — false content spread via network
        ("Act No. 3815", "Art. 315"),   # Estafa — fraudulent misrepresentation
    ],
    "reverse_image_hit": [
        ("RA 10173", "Sec. 25"),        # Unauthorized use/distribution of identifiable image
        ("RA 9995", "Sec. 4"),          # Photo/video voyeurism if intimate context
    ],
}

# Fact-check ratings that count as "debunked" for the enrichment path.
DEBUNKED_RATINGS = frozenset({
    "false", "mostly false", "misleading", "altered", "fake",
    "manipulated", "disinformation", "pants on fire",
})


def apply_mappings(
    report,
    tag_codes: list[str],
    *,
    face_count: int = 0,
    reverse_match_count: int = 0,
    debunked_ratings: list[str] | None = None,
) -> list[LegalMapping]:
    """
    Creates (or refreshes) LegalMapping rows for a DetectionReport.

    Parameters
    ----------
    report : DetectionReport
        The report to attach mappings to.
    tag_codes : list[str]
        Tag codes emitted by the Tagging Layer (e.g. ["identity_fraud"]).
    face_count : int
        Number of faces detected in the submission. When > 0 the
        "face_privacy_violation" enrichment path is triggered.
    reverse_match_count : int
        Number of reverse-image-search hits. When > 0 the
        "reverse_image_hit" enrichment path is triggered.
    debunked_ratings : list[str] | None
        Fact-check rating strings returned by the fact-check API.
        Any rating in DEBUNKED_RATINGS triggers the "debunked_fact_check" path.

    Returns
    -------
    list[LegalMapping]
        All LegalMapping rows created or fetched for this report.
    """
    # Build the full set of codes to evaluate, including enrichment sentinels.
    effective_codes: list[str] = list(tag_codes)

    if face_count > 0:
        effective_codes.append("face_privacy_violation")

    if reverse_match_count > 0:
        effective_codes.append("reverse_image_hit")

    if debunked_ratings:
        normalized = [r.lower().strip() for r in debunked_ratings]
        if any(r in DEBUNKED_RATINGS for r in normalized):
            effective_codes.append("debunked_fact_check")

    # Remove duplicates while preserving first-seen order.
    seen: set[str] = set()
    deduped: list[str] = []
    for code in effective_codes:
        if code not in seen:
            seen.add(code)
            deduped.append(code)

    # Wipe existing auto-mappings for this report so a re-run is idempotent.
    LegalMapping.objects.filter(
        report=report,
        trigger_condition__in=TAG_TO_PROVISIONS.keys(),
    ).delete()

    created: list[LegalMapping] = []
    for tag_code in deduped:
        provision_keys = TAG_TO_PROVISIONS.get(tag_code)
        if not provision_keys:
            # YOLO object labels (e.g. "person", "car") hit this path — skip silently.
            logger.debug("mapping_service: no provisions mapped for tag_code=%r", tag_code)
            continue

        for law_number, section in provision_keys:
            try:
                provision = LegalProvision.objects.get(law_number=law_number, section=section)
            except LegalProvision.DoesNotExist:
                logger.warning(
                    "mapping_service: LegalProvision not found — "
                    "law_number=%r section=%r (run seed_legal_provisions?)",
                    law_number,
                    section,
                )
                continue

            mapping, _ = LegalMapping.objects.get_or_create(
                report=report,
                provision=provision,
                trigger_condition=tag_code,
                defaults={
                    "notes": _build_note(tag_code, face_count, reverse_match_count, debunked_ratings),
                },
            )
            created.append(mapping)

    logger.info(
        "mapping_service: %d legal mapping(s) created for report_id=%s",
        len(created),
        report.pk,
    )
    return created


def generate_report_summary(report) -> dict:
    """
    Aggregates all LegalMapping rows for *report* into a structured summary
    dict. Called by legalmap/views.py to power the report API endpoint.

    Returns
    -------
    dict with keys:
        report_id          : int
        submission_id      : str (UUID)
        overall_verdict    : str
        confidence_score   : float
        summary_text       : str
        review_status      : str
        enacted_provisions : list[dict]   – provisions that are current law
        pending_provisions : list[dict]   – pending bills (must NOT be presented as law)
        risk_score         : float        – 0.0–1.0 composite risk indicator
        generated_at       : str (ISO-8601)
    """
    mappings = (
        LegalMapping.objects
        .filter(report=report)
        .select_related("provision")
        .order_by("provision__law_number", "provision__section")
    )

    enacted = []
    pending = []
    for m in mappings:
        p = m.provision
        entry = {
            "law_name": p.law_name,
            "law_number": p.law_number,
            "section": p.section,
            "description": p.description,
            "status": p.status,
            "trigger_condition": m.trigger_condition,
            "notes": m.notes,
            "flagged_at": m.flagged_at.isoformat(),
        }
        if p.status == "enacted":
            enacted.append(entry)
        else:
            pending.append(entry)

    # Simple risk score: base on confidence + provision count, capped at 1.0.
    base = float(report.confidence_score or 0.0)
    provision_factor = min(len(enacted) * 0.1, 0.4)   # up to +0.4 from enacted hits
    risk_score = round(min(base + provision_factor, 1.0), 4)

    return {
        "report_id": report.pk,
        "submission_id": str(report.submission_id),
        "overall_verdict": report.overall_verdict,
        "confidence_score": report.confidence_score,
        "summary_text": report.summary_text,
        "review_status": report.review_status,
        "enacted_provisions": enacted,
        "pending_provisions": pending,
        "risk_score": risk_score,
        "generated_at": report.generated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_note(
    tag_code: str,
    face_count: int,
    reverse_match_count: int,
    debunked_ratings: list[str] | None,
) -> str:
    if tag_code == "face_privacy_violation":
        return (
            f"Auto-mapped: {face_count} face(s) detected in submission. "
            "Biometric data processed without explicit consent signal — "
            "RA 10173 Sec. 25 and RA 9995 Sec. 4 apply."
        )
    if tag_code == "reverse_image_hit":
        return (
            f"Auto-mapped: {reverse_match_count} reverse-image-search match(es) found. "
            "Possible unauthorized redistribution of identifiable image."
        )
    if tag_code == "debunked_fact_check":
        ratings_str = ", ".join(debunked_ratings or [])
        return (
            f"Auto-mapped: fact-check API returned debunked rating(s): [{ratings_str}]. "
            "Potential cyberlibel or fraudulent misrepresentation exposure."
        )
    return f"Auto-mapped from tag '{tag_code}' by the Tagging Layer."