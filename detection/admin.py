from django.contrib import admin

from .models import (
    DetectionReport,
    ExifMetadata,
    FaceDetection,
    FactCheckReference,
    ForensicAnalysis,
    ImageSubmission,
    ReverseImageMatch,
)


class ExifMetadataInline(admin.StackedInline):
    model = ExifMetadata
    extra = 0


class FaceDetectionInline(admin.TabularInline):
    model = FaceDetection
    extra = 0


class ForensicAnalysisInline(admin.StackedInline):
    model = ForensicAnalysis
    extra = 0


class ReverseImageMatchInline(admin.TabularInline):
    model = ReverseImageMatch
    extra = 0


class FactCheckReferenceInline(admin.TabularInline):
    model = FactCheckReference
    extra = 0


@admin.register(ImageSubmission)
class ImageSubmissionAdmin(admin.ModelAdmin):
    list_display = ("id", "original_filename", "submitted_by", "status", "submitted_at")
    list_filter = ("status",)
    search_fields = ("original_filename", "sha256_hash")
    inlines = [
        ExifMetadataInline,
        FaceDetectionInline,
        ForensicAnalysisInline,
        ReverseImageMatchInline,
        FactCheckReferenceInline,
    ]


@admin.register(DetectionReport)
class DetectionReportAdmin(admin.ModelAdmin):
    list_display = ("submission", "overall_verdict", "confidence_score", "review_status")
    list_filter = ("review_status", "overall_verdict")
