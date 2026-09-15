from django.db import models

from detection.models import DetectionReport


class LegalProvision(models.Model):
    """A specific Philippine legal provision relevant to synthetic/manipulated media."""
    
    class Status(models.TextChoices):
        ENACTED = "enacted", "Enacted Law"
        PENDING_BILL = "pending_bill", "Pending Bill (not yet law)"

    law_name = models.CharField(
        max_length=255, help_text="e.g. 'Cybercrime Prevention Act of 2012'"
    )
    law_number = models.CharField(max_length=50, help_text="e.g. 'RA 10175'")
    section = models.CharField(max_length=50, blank=True, help_text="e.g. 'Sec. 4(b)(3)'")
    description = models.TextField()
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ENACTED,
        help_text="Whether this is currently enacted law or a pending bill — do not present pending bills as current law.",
    )
    jurisdiction = models.CharField(max_length=100, default="Philippines")

    class Meta:
        unique_together = ("law_number", "section")
        ordering = ["law_number", "section"]

    def __str__(self):
        return f"{self.law_number} {self.section} - {self.law_name}"


class LegalMapping(models.Model):
    """
    Links a detection report's outcome to the legal provisions it may trigger.
    This is the layer your panel flagged during the defense — keep the
    trigger_condition values enumerated and reviewed by legal counsel, not
    inferred silently by the model.
    """

    report = models.ForeignKey(
        DetectionReport, on_delete=models.CASCADE, related_name="legal_mappings"
    )
    provision = models.ForeignKey(
        LegalProvision, on_delete=models.PROTECT, related_name="mappings"
    )
    trigger_condition = models.CharField(
        max_length=100,
        help_text="e.g. 'manipulated_verdict_high_confidence', 'face_swap_of_public_official'",
    )
    notes = models.TextField(blank=True)
    flagged_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("report", "provision", "trigger_condition")

    def __str__(self):
        return f"{self.report_id} -> {self.provision.law_number}"
