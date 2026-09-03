import logging
from celery import shared_task
from django.contrib.auth import get_user_model

from detection.pipeline import process_submission

User = get_user_model()
logger = logging.getLogger(__name__)


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