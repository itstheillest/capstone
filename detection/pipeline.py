"""
Orchestrates the full detection pipeline for one ImageSubmission:
EXIF check -> face detection -> forensic (ELA baseline) analysis -> status update.

Each step writes its own AuditLog entry, and any failure marks the
submission FAILED with the error logged rather than raising silently past
the caller.
"""

from django.utils import timezone

from audit.models import AuditLog
from detection.models import ExifMetadata, FaceDetection, ForensicAnalysis, ImageSubmission
from detection.models import (
    DetectionReport,
    ExifMetadata,
    FaceDetection,
    ForensicAnalysis,
    ImageSubmission,
    TaggingResult,
)
from detection.services import exif_service, face_service, forensic_service, tagging_service
from legalmap.services import mapping_service


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
        # --- EXIF ---
        exif_data = exif_service.extract_exif(image_path)
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

        # --- Face detection ---
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

        # --- Forensic analysis ---
        forensic_result = forensic_service.analyze(image_path)
        ForensicAnalysis.objects.update_or_create(
            submission=submission,
            defaults={
                "model_name": forensic_result["model_name"],
                "model_version": forensic_result["model_version"],
                "manipulation_score": forensic_result["manipulation_score"],
                "verdict": forensic_result["verdict"],
                "anomaly_regions": forensic_result["anomaly_regions"],
            },
        )
        
        _log(
            submission,
            AuditLog.ActionType.FORENSIC_ANALYSIS,
            {"verdict": forensic_result["verdict"], "score": forensic_result["manipulation_score"]},
            actor,
        )
        # --- Tagging Layer ---
        TaggingResult.objects.filter(submission=submission).delete()
        tags = tagging_service.generate_tags(
            face_count=len(faces),
            verdict=forensic_result["verdict"],
            manipulation_score=forensic_result["manipulation_score"],
        )
        for tag in tags:
            TaggingResult.objects.create(submission=submission, **tag)
        _log(
            submission,
            AuditLog.ActionType.TAGGING,
            {"tags": [t["tag_code"] for t in tags]},
            actor,
        )
        
        # --- Minimal auto-generated report + Law Mapping Layer ---
        tag_summary = ", ".join(t["tag_label"] for t in tags) if tags else "No tags generated."
        report, _ = DetectionReport.objects.update_or_create(
            submission=submission,
            defaults={
                "overall_verdict": forensic_result["verdict"],
                "confidence_score": forensic_result["manipulation_score"],
                "summary_text": (
                    f"Automated analysis: forensic verdict '{forensic_result['verdict']}' "
                    f"(score {forensic_result['manipulation_score']:.3f}), "
                    f"{len(faces)} face(s) detected. Tags: {tag_summary}"
                ),
                "review_status": DetectionReport.ReviewStatus.AUTO_GENERATED,
            },
        )
        legal_mappings = mapping_service.apply_mappings(
            report, [t["tag_code"] for t in tags]
        )
        _log(
            submission,
            AuditLog.ActionType.REPORT_GENERATED,
            {"legal_provisions_mapped": [str(m.provision) for m in legal_mappings]},
            actor,
        )
        
        submission.status = ImageSubmission.Status.COMPLETED
        submission.completed_at = timezone.now()
        submission.save(update_fields=["status", "completed_at"])

    except Exception as exc:
        submission.status = ImageSubmission.Status.FAILED
        submission.save(update_fields=["status"])
        _log(submission, AuditLog.ActionType.FORENSIC_ANALYSIS, {"error": str(exc)}, actor)
        raise

    return submission