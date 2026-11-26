"""
Orchestrates the full detection pipeline for one ImageSubmission:
EXIF check -> Face detection -> Forensic analysis -> Tagging -> Report & Legal mapping -> Fact check & Reverse search -> Status update.

Each step writes its own AuditLog entry. Any failure marks the submission
FAILED with the error logged.
"""
import os
import logging
from django.db import transaction
from django.utils import timezone

from audit.models import AuditLog
from detection.models import (
    DetectionReport,
    ExifMetadata,
    FaceDetection,
    FactCheckReference,
    ForensicAnalysis,
    ImageSubmission,
    ReverseImageMatch,
    TaggingResult,
)

from detection.services import (
    exif_service,
    face_service,
    forensic_service,
    tagging_service,
    teachable_machine_service,
)

from detection.services.factcheck_service import FactCheckService
from legalmap.services import mapping_service
from detection.services import ocr_service

logger = logging.getLogger(__name__)

_VALID_MATCH_TYPES = {"EXACT", "PARTIAL", "SIMILAR"}


def resolve_final_verdict(
    visual_verdict: str,
    provenance_status: str,
) -> tuple[str, str]:
    """Return (final verdict, reason) using the visual result and provenance."""
    if provenance_status == "verified_ai":
        return "manipulated", "verified_ai_provenance"

    if provenance_status == "unverified_ai_marker":
        return "inconclusive", "unverified_ai_marker"

    return visual_verdict, "visual_classifier"

def _log(submission, action_type, details=None, actor=None):
    AuditLog.objects.create(
        submission=submission,
        actor=actor,
        action_type=action_type,
        details=details or {},
    )



def process_submission(submission_id: str, actor=None) -> ImageSubmission:
    submission = ImageSubmission.objects.get(id=submission_id)
    submission.status = ImageSubmission.Status.PROCESSING
    submission.save(update_fields=["status"])

    image_path = submission.image.path
    try:
        ocr_result = ocr_service.extract_text(image_path)
        submission.ocr_status = ocr_result["status"]
        submission.extracted_text = ocr_result["text"]
        submission.ocr_confidence = ocr_result["confidence"]
        submission.save(
            update_fields=["ocr_status", "extracted_text", "ocr_confidence"]
        )

        # --- 1. EXIF Analysis ---
        exif_data = exif_service.extract_exif(image_path)
        
        provenance_status = exif_data.get("c2pa_status", "none")

        if (provenance_status in {"none", "validation_error"} and exif_data.get("ai_metadata_detected")):
            provenance_status = "unverified_ai_marker"
        
        ExifMetadata.objects.update_or_create(
            submission=submission,
            defaults={
                "raw_exif": exif_data["raw_exif"],
                "camera_make": exif_data["camera_make"],
                "camera_model": exif_data["camera_model"],
                "software_tag": exif_data["software_tag"],
                "capture_timestamp": exif_data["capture_timestamp"],
                "gps_latitude": exif_data["gps_latitude"],
                "gps_longitude": exif_data["gps_longitude"],
                "has_discrepancy": exif_data["has_discrepancy"],
                "discrepancy_notes": exif_data["discrepancy_notes"],
            },
        )
        _log(
            submission,
            AuditLog.ActionType.EXIF_CHECK,
            {"has_discrepancy": exif_data["has_discrepancy"], "used_exiftool": exif_data["used_exiftool"]},
            actor,
        )

        # --- 2. Face Detection ---
        FaceDetection.objects.filter(submission=submission).delete()
        faces = face_service.detect_faces(image_path)
        for i, face in enumerate(faces):
            FaceDetection.objects.create(
                submission=submission,
                face_index=i,
                bounding_box=face["bounding_box"],
                detection_confidence=face["detection_confidence"],
            )
        _log(submission, AuditLog.ActionType.FACE_DETECT, {"face_count": len(faces)}, actor)

        # --- 3. Forensic Analysis ---
        forensic_result = forensic_service.analyze(image_path)
        teachable_result = None
        
        visual_verdict = forensic_result["verdict"]
        final_verdict, decision_basis = resolve_final_verdict(visual_verdict, provenance_status,)
        
        if os.getenv("TEACHABLE_MACHINE_ENABLED", "false").lower() == "true":
            try:
                teachable_result = teachable_machine_service.predict(image_path)
            except Exception:
                logger.exception(
                    "Experimental Teachable Machine inference failed for submission %s",
                    submission.id,
                )
                teachable_result = {
                    "model": "Teachable Machine image classifier",
                    "version": "tm-export-1",
                    "status": "error",
                    "available": False,
                }
        
        ForensicAnalysis.objects.update_or_create(
            submission=submission,
            defaults={
                "model_name": forensic_result["model_name"],
                "model_version": forensic_result["model_version"],
                "manipulation_score": forensic_result["manipulation_score"],
                "verdict": visual_verdict,
                "anomaly_regions": forensic_result["anomaly_regions"],
                "processing_time_ms": forensic_result.get("processing_time_ms"),
            },
        )
        
        _log(
            submission,
            AuditLog.ActionType.FORENSIC_ANALYSIS,
            {
                "verdict": visual_verdict,
                "score": forensic_result["manipulation_score"],
                "model": forensic_result["model_name"],
                "processing_time_ms": forensic_result.get("processing_time_ms"),
                "teachable_machine": teachable_result,

            },
            actor,
        )


        # --- 4. Tagging Layer ---
        TaggingResult.objects.filter(submission=submission).delete()
        
        tags = tagging_service.generate_tags(
            image_path=image_path,
            face_count=len(faces),
            verdict=final_verdict,
            manipulation_score=forensic_result["manipulation_score"],
            has_exif_discrepancy=exif_data["has_discrepancy"],
            ai_metadata_detected=exif_data["ai_metadata_detected"],
            ai_metadata_markers=exif_data["ai_metadata_markers"],
            provenance_status=provenance_status,
        )
        
        for tag in tags:
            TaggingResult.objects.create(submission=submission, **tag)
        _log(
            submission,
            AuditLog.ActionType.TAGGING,
            {"tag_codes": [t["tag_code"] for t in tags]},
            actor,
        )

        # --- 5. Report & Law Mapping Layer ---
        tag_summary = ", ".join(t["tag_label"] for t in tags) if tags else "No tags generated."
        report, _ = DetectionReport.objects.update_or_create(
            submission=submission,
           defaults={
                "overall_verdict": final_verdict,
                "confidence_score": forensic_result["manipulation_score"],
                "summary_text": (
                    f"Final verdict: '{final_verdict}' "
                    f"(decision basis: {decision_basis}). "
                    f"Visual classifier verdict: '{visual_verdict}' "
                    f"(score {forensic_result['manipulation_score']:.3f}). "
                    f"{len(faces)} face(s) detected. Tags: {tag_summary}"
                ),
                "review_status": DetectionReport.ReviewStatus.AUTO_GENERATED,
            },
        )
        
        # Initial legal mapping — runs immediately after tagging, before fact-check.
        # A second enriched pass runs after the fact-check layer below.
        face_count = len(faces)
        legal_mappings = mapping_service.apply_mappings(
            report,
            [t["tag_code"] for t in tags],
            face_count=face_count,
        )
        _log(
            submission,
            AuditLog.ActionType.REPORT_GENERATED,
            {"legal_provisions_mapped": [str(m.provision) for m in legal_mappings]},
            actor,
        )

        # --- 6. Fact Checking & Reverse Search Layer ---
        context_query = ocr_result["text"] or " ".join(t["tag_label"] for t in tags) if tags else ocr_result["text"]
        process_fact_checking_layer(
            submission=submission,
            report=report,
            image_path=image_path,
            context_query=context_query,
            actor=actor,
        )

        # --- 6b. Re-run legal mapping enriched with fact-check + reverse search results ---
        reverse_match_count = submission.reverse_matches.count()
        debunked_ratings = list(
            submission.fact_check_references.values_list("rating", flat=True)
        )
        mapping_service.apply_mappings(
            report,
            [t["tag_code"] for t in tags],
            face_count=face_count,
            reverse_match_count=reverse_match_count,
            debunked_ratings=debunked_ratings,
        )

        # --- Completion ---
        submission.status = ImageSubmission.Status.COMPLETED
        submission.completed_at = timezone.now()
        submission.save(update_fields=["status", "completed_at"])

    except Exception as exc:
        submission.status = ImageSubmission.Status.FAILED
        submission.save(update_fields=["status"])
        _log(submission, AuditLog.ActionType.FORENSIC_ANALYSIS, {"error": str(exc)}, actor)
        raise

    return submission


def process_fact_checking_layer(
    submission: ImageSubmission,
    report=None,
    image_path: str = "",
    context_query: str = "",
    actor=None,
):
    """
    Executes Reverse Search and Fact-Check Lookup, saving results idempotently to DB.
    """
    # 1. Reverse Image Search
    ReverseImageMatch.objects.filter(submission=submission).delete()
    matches = FactCheckService.perform_reverse_image_search(image_path)
    if matches:
        unique_matches = {
            match["page_url"]: match
            for match in matches
            if match.get("page_url")
        }
        with transaction.atomic():
            submission.reverse_matches.all().delete()
            ReverseImageMatch.objects.bulk_create([
                ReverseImageMatch(
                    submission=submission,
                    page_url=match["page_url"],
                    page_title=(match.get("page_title") or "")[:512],
                    image_url=match.get("image_url", ""),
                    domain=(match.get("domain", "") or "")[:255],
                    similarity_score=match.get("similarity_score", 0.0),
                    match_type=match.get("match_type", "UNKNOWN"),
                )
                for match in unique_matches.values()
            ])

    _log(
        submission,
        AuditLog.ActionType.REVERSE_SEARCH,
        {"matches_found": len(matches)},
        actor,
    )

    # 2. Fact Check Lookup
    search_query = context_query
    lookup = {"status": "not_searched", "references": []}

    if search_query and search_query.strip():
        lookup = FactCheckService.query_fact_check_tools(search_query)

    if lookup["status"] in {"found", "no_results"}:
        _replace_fact_check_references(submission, lookup["references"])

    # 3. Log Audit Action
    action_type = getattr(AuditLog.ActionType, "FACT_CHECK", "FACT_CHECK")
    _log(
        submission,
        action_type,
        {
            "matches_found": len(matches),
            "fact_checks_found": len(lookup["references"]),
            "fact_check_status": lookup["status"],
        },
        actor,
    )


def _replace_fact_check_references(submission, references):
    unique_references = {}
    
    for reference in references:
        publisher_url = reference.get("publisher_url", "")
        if publisher_url.startswith(("https://", "http://")):
            unique_references[(publisher_url, reference.get("claim_text", ""))] = reference

    rating=(reference.get("rating") or "Unrated").strip()[:100],
    
    with transaction.atomic():
        FactCheckReference.objects.filter(submission=submission).delete()
        FactCheckReference.objects.bulk_create([
            FactCheckReference(
                submission=submission,
                claim_text=reference.get("claim_text", ""),
                claimant=reference.get("claimant", ""),
                publisher_name=reference.get("publisher_name", "Unknown"),
                publisher_url=reference["publisher_url"],
                rating=reference.get("rating", "Unrated"),
                review_date=reference.get("review_date") or timezone.now().date(),
            )
            for reference in unique_references.values()
        ])


def retry_fact_check_lookup(submission_id: str) -> str:
    """Retry only the fact-check lookup using the saved OCR text."""
    submission = ImageSubmission.objects.get(pk=submission_id)
    query_text = submission.extracted_text.strip()

    if not query_text:
        lookup = {"status": "not_searched", "references": []}
    else:
        lookup = FactCheckService.query_fact_check_tools(query_text)

    # Replace results only after a completed lookup. If the API is still
    # unavailable, keep any previously successful results intact.
    with transaction.atomic():
        if lookup["status"] in {"found", "no_results"}:
            _replace_fact_check_references(submission, lookup["references"])

        _log(
            submission,
            AuditLog.ActionType.FACT_CHECK,
            {
                "matches_found": submission.reverse_matches.count(),
                "fact_checks_found": len(lookup["references"]),
                "fact_check_status": lookup["status"],
                "retry": True,
            },
        )

    return lookup["status"]


def trigger_image_pipeline(submission_id: str, async_execution: bool = True):
    """
    Pipeline entry point to dispatch processing for an ImageSubmission.
    """
    try:
        ImageSubmission.objects.get(pk=submission_id)
    except ImageSubmission.DoesNotExist:
        logger.error("Cannot run pipeline: Submission %s not found.", submission_id)
        return False

    if async_execution:
        from detection.tasks import process_submission_task
        transaction.on_commit(lambda: process_submission_task.delay(str(submission_id)))
    else:
        process_submission(str(submission_id))

    return True
