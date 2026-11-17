from django.core.management.base import BaseCommand

from detection.models import ImageSubmission


class Command(BaseCommand):
    help = "Re-run the analysis pipeline on existing submissions (keeps images and audit logs)."

    def add_arguments(self, parser):
        parser.add_argument("--sync", action="store_true",
                            help="run in this process instead of queueing to the Celery worker")

    def handle(self, *args, **opts):
        ids = list(ImageSubmission.objects.values_list("id", flat=True))
        if opts["sync"]:
            from detection.pipeline import process_submission

            ok = 0
            for sid in ids:
                try:
                    process_submission(str(sid))
                    ok += 1
                except Exception as exc:
                    self.stderr.write(f"FAILED {sid}: {exc}")
            self.stdout.write(self.style.SUCCESS(f"Reprocessed {ok}/{len(ids)} submissions."))
        else:
            from detection.tasks import process_submission_task

            for sid in ids:
                process_submission_task.delay(str(sid))
            self.stdout.write(self.style.SUCCESS(f"Queued {len(ids)} submissions for the Celery worker."))