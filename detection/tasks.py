import logging
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.utils import timezone
from celery import shared_task

from audit.models import AuditLog
from detection.models import ImageSubmission
from detection.pipeline import process_submission, retry_fact_check_lookup

User = get_user_model()
logger = logging.getLogger(__name__)

def discard_submission_image(submission):
    """
    Delete the uploaded source image and clear its database file reference.
    Keep the ImageSubmission and its analysis records.
    """
    if not submission.image or not submission.image.name:
        return

    submission.image.delete(save=False)
    submission.image = ""
    submission.save(update_fields=["image"])


@shared_task
def cleanup_old_media_files(max_age_seconds=86400):
    """Back up immediate cleanup for old completed or failed submissions."""
    cutoff = timezone.now() - timedelta(seconds=max_age_seconds)
    old_submissions = ImageSubmission.objects.filter(
        submitted_at__lt=cutoff,
        status__in=[
            ImageSubmission.Status.COMPLETED,
            ImageSubmission.Status.FAILED,
        ],
    ).exclude(image="")

    for submission in old_submissions.iterator():
        try:
            discard_submission_image(submission)
        except Exception:
            logger.exception(
                "Could not clean old image for submission %s.",
                submission.id,
            )


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
    actor = None
    if actor_id:
        try:
            actor = User.objects.get(id=actor_id)
        except User.DoesNotExist:
            logger.warning("Actor with ID %s not found for task execution.", actor_id)

    try:
        # This runs all steps that need the image file.
        submission = process_submission(submission_id, actor=actor)

    except Exception as exc:
        logger.exception(
            "Error executing process_submission_task for %s.",
            submission_id,
        )

        # Keep the image available while Celery may still retry.
        # On the final failed attempt, remove it.
        if self.request.retries >= self.max_retries:
            failed_submission = ImageSubmission.objects.filter(
                pk=submission_id
            ).first()

            if failed_submission:
                try:
                    discard_submission_image(failed_submission)
                except Exception:
                    logger.exception(
                        "Could not delete uploaded image for failed submission %s.",
                        submission_id,
                    )

            raise

        raise self.retry(exc=exc)

    # Processing succeeded; now it is safe to remove the source image.
    try:
        discard_submission_image(submission)
    except Exception:
        # The cleanup task below is a backup in case immediate deletion fails.
        logger.exception(
            "Analysis completed, but uploaded image cleanup failed for %s.",
            submission_id,
        )

    return str(submission.id)
