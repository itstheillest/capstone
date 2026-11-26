import os
from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pixifai.settings")

app = Celery("pixifai")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

app.conf.beat_schedule = {
    "cleanup-old-submission-images-daily": {
        "task": "detection.tasks.cleanup_old_media_files",
        "schedule": crontab(hour=3, minute=0),
        "kwargs": {"max_age_seconds": 86400},
    },
}
