import os
import time
import logging

from django.contrib.auth import get_user_model
from django.conf import settings
from celery import shared_task

from audit.models import AuditLog
from detection.models import ImageSubmission
from detection.pipeline import process_submission, retry_fact_check_lookup

User = get_user_model()
logger = logging.getLogger(__name__)

@shared_task
def retry_fact_check_task(submission_id: str):
    try:
        return retry_fact_check_lookup(submission_id)
    except Exception:
        logger.exception(
            "Fact-check-only retry failed for submission %s.",
            submission_id,
        )
        try:
            submission = ImageSubmission.objects.get(pk=submission_id)
            AuditLog.objects.create(
                submission=submission,
                action_type=AuditLog.ActionType.FACT_CHECK,
                details={
                    "fact_check_status": "unavailable",
                    "fact_checks_found": submission.fact_check_references.count(),
                    "retry": True,
                    "error": "Unexpected retry failure",
                },
            )
        except ImageSubmission.DoesNotExist:
            logger.warning(
                "Cannot record retry failure; submission %s was not found.",
                submission_id,
            )
        return "unavailable"

@shared_task(bind=True, max_retries=3, default_retry_delay=15)
def process_submission_task(self, submission_id: str, actor_id: str | None = None):
    """
    Async Celery task to process an ImageSubmission end-to-end.
    Delegates all analysis (EXIF, Face, Forensic, Tags, Legal, Fact Check)
    to detection.pipeline.process_submission.
    """
    actor = None
    if actor_id:
        try:
            actor = User.objects.get(id=actor_id)
        except User.DoesNotExist:
            logger.warning(f"Actor with ID {actor_id} not found for task execution.")

    try:
        submission = process_submission(submission_id, actor=actor)
        return str(submission.id)
    except Exception as exc:
        logger.exception(f"Error executing process_submission_task for {submission_id}: {exc}")
        raise self.retry(exc=exc)
    
@shared_task
def cleanup_old_media_files(max_age_seconds=86400):
    """Deletes uploaded media files older than max_age_seconds (default 24h)."""
    now = time.time()
    media_dir = settings.MEDIA_ROOT

    for root, _, files in os.walk(media_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) < (now - max_age_seconds):
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.error(f"Failed deleting old media file {file_path}: {e}")
