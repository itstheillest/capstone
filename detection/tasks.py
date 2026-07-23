from celery import shared_task

from accounts.models import User
from detection.pipeline import process_submission


@shared_task(bind=True, max_retries=2, default_retry_delay=30)
def process_submission_task(self, submission_id: str, actor_id: str | None = None):
    """
    Async wrapper around detection.pipeline.process_submission. Call this
    from the upload view with .delay(submission.id) instead of calling
    process_submission directly, so the HTTP request returns immediately
    while EXIF/face/forensic processing runs in a Celery worker.
    """
    actor = None
    if actor_id:
        try:
            actor = User.objects.get(id=actor_id)
        except User.DoesNotExist:
            pass

    try:
        submission = process_submission(submission_id, actor=actor)
        return str(submission.id)
    except Exception as exc:
        raise self.retry(exc=exc)