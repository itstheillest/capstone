import uuid

from django.conf import settings
from django.db import models


def submission_upload_path(instance, filename):
    return f"submissions/{instance.id}/{filename}"


def heatmap_upload_path(instance, filename):
    return f"heatmaps/{instance.submission_id}/{filename}"


class ImageSubmission(models.Model):
    """One uploaded image and its processing lifecycle."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    submitted_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="submissions",
    )
    image = models.ImageField(upload_to=submission_upload_path)
    original_filename = models.CharField(max_length=255)
    sha256_hash = models.CharField(max_length=64, unique=True, db_index=True)
    file_size_bytes = models.PositiveIntegerField()
    mime_type = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    submitted_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-submitted_at"]
        indexes = [models.Index(fields=["status", "submitted_at"])]

    def __str__(self):
        return f"{self.original_filename} ({self.status})"


class ExifMetadata(models.Model):
    """Metadata forensics pulled via ExifTool."""

    submission = models.OneToOneField(
        ImageSubmission, on_delete=models.CASCADE, related_name="exif"
    )
    raw_exif = models.JSONField(default=dict, help_text="Full raw ExifTool output.")
    camera_make = models.CharField(max_length=100, blank=True)
    camera_model = models.CharField(max_length=100, blank=True)
    software_tag = models.CharField(
        max_length=150, blank=True, help_text="Editing software tag, e.g. 'Adobe Photoshop 25.0'"
    )
    capture_timestamp = models.DateTimeField(null=True, blank=True)
    gps_latitude = models.FloatField(null=True, blank=True)
    gps_longitude = models.FloatField(null=True, blank=True)
    has_discrepancy = models.BooleanField(
        default=False,
        help_text="True if EXIF is missing/stripped/inconsistent with claimed provenance.",
    )
    discrepancy_notes = models.TextField(blank=True)

    def __str__(self):
        return f"EXIF for {self.submission_id}"


class FaceDetection(models.Model):
    """One detected face within a submission (face_recognition / DeepFace)."""

    submission = models.ForeignKey(
        ImageSubmission, on_delete=models.CASCADE, related_name="faces"
    )
    face_index = models.PositiveSmallIntegerField()
    bounding_box = models.JSONField(help_text="[top, right, bottom, left] in pixels.")
    detection_confidence = models.FloatField()
    embedding = models.JSONField(
        null=True, blank=True, help_text="128/512-d face embedding vector for matching."
    )
    deepface_emotion = models.CharField(max_length=50, blank=True)
    deepface_age_estimate = models.PositiveSmallIntegerField(null=True, blank=True)

    class Meta:
        unique_together = ("submission", "face_index")

    def __str__(self):
        return f"Face {self.face_index} in {self.submission_id}"


class ForensicAnalysis(models.Model):
    """Output of the CNN/ViT deepfake classifier + YOLOv8/Detectron2 region flags."""

    class Verdict(models.TextChoices):
        AUTHENTIC = "authentic", "Likely Authentic"
        MANIPULATED = "manipulated", "Likely Manipulated"
        INCONCLUSIVE = "inconclusive", "Inconclusive"

    submission = models.OneToOneField(
        ImageSubmission, on_delete=models.CASCADE, related_name="forensic_analysis"
    )
    model_name = models.CharField(max_length=100, help_text="e.g. ViT-DeepfakeDetector-v1")
    model_version = models.CharField(max_length=50)
    manipulation_score = models.FloatField(help_text="0.0 (authentic) to 1.0 (manipulated).")
    verdict = models.CharField(max_length=20, choices=Verdict.choices)
    anomaly_regions = models.JSONField(
        default=list,
        help_text="List of {box, label, confidence} from YOLOv8/Detectron2 region flags.",
    )
    heatmap_image = models.ImageField(
        upload_to=heatmap_upload_path, null=True, blank=True, help_text="Grad-CAM overlay."
    )
    processing_time_ms = models.PositiveIntegerField(null=True, blank=True)
    analyzed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.verdict} ({self.manipulation_score:.2f}) - {self.submission_id}"


class ReverseImageMatch(models.Model):
    """A single external match found via reverse image search."""

    submission = models.ForeignKey(
        ImageSubmission, on_delete=models.CASCADE, related_name="reverse_matches"
    )
    source_url = models.URLField(max_length=500)
    source_domain = models.CharField(max_length=255, blank=True)
    match_confidence = models.FloatField(null=True, blank=True)
    first_indexed_date = models.DateField(null=True, blank=True)
    thumbnail_url = models.URLField(max_length=500, blank=True)

    def __str__(self):
        return f"{self.source_domain} match for {self.submission_id}"


class FactCheckReference(models.Model):
    """A ClaimReview result pulled from Google Fact Check Tools API."""

    submission = models.ForeignKey(
        ImageSubmission, on_delete=models.CASCADE, related_name="fact_checks"
    )
    claim_text = models.TextField()
    claim_review_url = models.URLField(max_length=500)
    publisher_name = models.CharField(max_length=255, blank=True)
    rating_text = models.CharField(max_length=100, blank=True)
    rating_date = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"Fact-check for {self.submission_id}: {self.rating_text}"

class TaggingResult(models.Model):
    """
    Output of the Tagging Layer: rule-based combination of what the
    YOLOv8 object detector found (objects of interest, e.g. faces) with the
    Synthetic Media Analysis Layer's verdict, producing a semantic tag
    (e.g. 'identity_fraud'). LegalMapping then looks up which Philippine
    laws a given tag_code may implicate.
    """

    submission = models.ForeignKey(
        ImageSubmission, on_delete=models.CASCADE, related_name="tags"
    )
    tag_code = models.CharField(
        max_length=50, help_text="Machine-readable code, e.g. 'identity_fraud'. Used as the LegalMapping lookup key."
    )
    tag_label = models.CharField(max_length=150, help_text="Human-readable label shown in the UI.")
    rule_description = models.TextField(
        help_text="Explains which detections triggered this tag, for auditability."
    )
    confidence = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.tag_code} ({self.submission_id})"

class DetectionReport(models.Model):
    """Final aggregated report shown to the requesting agency."""

    class ReviewStatus(models.TextChoices):
        AUTO_GENERATED = "auto_generated", "Auto-generated (unreviewed)"
        HUMAN_REVIEWED = "human_reviewed", "Human-reviewed"
        DISPUTED = "disputed", "Disputed"

    submission = models.OneToOneField(
        ImageSubmission, on_delete=models.CASCADE, related_name="report"
    )
    overall_verdict = models.CharField(max_length=20)
    confidence_score = models.FloatField()
    summary_text = models.TextField()
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    review_status = models.CharField(
        max_length=20, choices=ReviewStatus.choices, default=ReviewStatus.AUTO_GENERATED
    )
    generated_pdf = models.FileField(upload_to="reports/", null=True, blank=True)
    generated_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Report for {self.submission_id} ({self.review_status})"
