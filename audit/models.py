from django.conf import settings
from django.db import models

from detection.models import ImageSubmission


class AuditLog(models.Model):
    """
    Immutable trail of every action taken on a submission — required for
    chain-of-custody if a report is ever used as evidence or cited officially.
    Never update or delete rows here; only append.
    """

    class ActionType(models.TextChoices):
        UPLOAD = "upload", "Image Uploaded"
        EXIF_CHECK = "exif_check", "EXIF Metadata Checked"
        FACE_DETECT = "face_detect", "Face Detection Run"
        FORENSIC_ANALYSIS = "forensic_analysis", "Forensic Analysis Run"
        TAGGING = "tagging", "Tagging Layer Run"
        REVERSE_SEARCH = "reverse_search", "Reverse Image Search Run"
        FACT_CHECK = "fact_check", "Fact-Check Lookup Run"
        REPORT_GENERATED = "report_generated", "Report Generated"
        REPORT_REVIEWED = "report_reviewed", "Report Reviewed by Human"
        EXPORT = "export", "Report Exported"
        

    submission = models.ForeignKey(
        ImageSubmission, on_delete=models.SET_NULL, null=True, related_name="audit_logs"
    )
    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="Null when the action was performed by an automated pipeline step.",
    )
    action_type = models.CharField(max_length=30, choices=ActionType.choices)
    details = models.JSONField(default=dict, blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes = [models.Index(fields=["submission", "timestamp"])]

    def __str__(self):
        return f"[{self.timestamp}] {self.action_type} on {self.submission_id}"
