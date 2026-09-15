from django.contrib import admin
from .models import ImageSubmission, ReverseImageMatch, FactCheckReference, DetectionReport


class ReverseImageMatchInline(admin.TabularInline):
    model = ReverseImageMatch
    extra = 0
    readonly_fields = ["page_url", "domain", "match_type", "similarity_score", "created_at"]


class FactCheckReferenceInline(admin.TabularInline):
    model = FactCheckReference
    extra = 0
    readonly_fields = ["claim_text", "publisher_name", "publisher_url", "rating", "review_date"]


@admin.register(ImageSubmission)
class ImageSubmissionAdmin(admin.ModelAdmin):
    list_display = ["id", "original_filename", "status", "submitted_at", "completed_at"]
    list_filter = ["status", "submitted_at"]
    search_fields = ["id", "sha256_hash", "original_filename"]
    inlines = [ReverseImageMatchInline, FactCheckReferenceInline]