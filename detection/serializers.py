from rest_framework import serializers
from .models import ImageSubmission, ReverseImageMatch, FactCheckReference, DetectionReport


class ReverseImageMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReverseImageMatch
        fields = ["id", "page_url", "domain", "match_type", "similarity_score"]


class FactCheckReferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FactCheckReference
        fields = ["id", "claim_text", "publisher_name", "publisher_url", "rating", "review_date"]


class DetectionReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionReport
        fields = ["id", "overall_verdict", "confidence_score", "summary_text", "review_status"]


class ImageSubmissionSerializer(serializers.ModelSerializer):
    reverse_matches = ReverseImageMatchSerializer(many=True, read_only=True)
    fact_check_references = FactCheckReferenceSerializer(many=True, read_only=True)
    report = DetectionReportSerializer(read_only=True)

    class Meta:
        model = ImageSubmission
        fields = [
            "id",
            "image",
            "sha256_hash",
            "original_filename",
            "file_size_bytes",
            "mime_type",
            "status",
            "submitted_at",
            "completed_at",
            "report",
            "reverse_matches",
            "fact_check_references",
        ]
        read_only_fields = [
            "id",
            "sha256_hash",
            "original_filename",
            "file_size_bytes",
            "mime_type",
            "status",
            "submitted_at",
            "completed_at",
        ]