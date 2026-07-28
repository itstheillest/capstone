'''
from rest_framework import serializers
from legalmap.serializers import LegalMappingSerializer

from .models import (
    DetectionReport,
    ExifMetadata,
    FaceDetection,
    FactCheckReference,
    ForensicAnalysis,
    ImageSubmission,
    ReverseImageMatch,
    TaggingResult,
)
from rest_framework import serializers
from detection.models import ReverseImageMatch, FactCheckReference, DetectionReport

class TaggingResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaggingResult
        exclude = ["submission"]


class ImageSubmissionUploadSerializer(serializers.ModelSerializer):
    """Write-only serializer used for POST /api/submissions/."""

    class Meta:
        model = ImageSubmission
        fields = ["id", "image", "status", "submitted_at"]
        read_only_fields = ["id", "status", "submitted_at"]

    def validate_image(self, value):
        max_size_mb = 15
        if value.size > max_size_mb * 1024 * 1024:
            raise serializers.ValidationError(f"Image must be under {max_size_mb}MB.")

        allowed_types = {"image/jpeg", "image/png", "image/webp"}
        content_type = getattr(value, "content_type", None)
        if content_type and content_type not in allowed_types:
            raise serializers.ValidationError(
                f"Unsupported file type '{content_type}'. Allowed: JPEG, PNG, WEBP."
            )
        return value


class ImageSubmissionListSerializer(serializers.ModelSerializer):
    """Lightweight serializer used for GET /api/submissions/ (list view)."""

    submitted_by = serializers.StringRelatedField()

    class Meta:
        model = ImageSubmission
        fields = [
            "id",
            "original_filename",
            "status",
            "submitted_by",
            "submitted_at",
            "completed_at",
        ]


class ExifMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExifMetadata
        exclude = ["id", "submission"]


class FaceDetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceDetection
        exclude = ["submission"]


class ForensicAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForensicAnalysis
        exclude = ["id", "submission"]


class ReverseImageMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReverseImageMatch
        exclude = ["submission"]


class FactCheckReferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FactCheckReference
        exclude = ["submission"]


class DetectionReportSerializer(serializers.ModelSerializer):
    reviewed_by = serializers.StringRelatedField()
    legal_mappings = LegalMappingSerializer(many=True, read_only=True)

    class Meta:
        model = DetectionReport
        exclude = ["id", "submission"]


class ImageSubmissionDetailSerializer(serializers.ModelSerializer):
    """
    Full nested serializer used for GET /api/submissions/{id}/.
    Pulls in every pipeline stage that has run so far; fields will simply be
    null/empty until each stage (EXIF, faces, forensic analysis, etc.) has
    actually processed this submission.
    """

    submitted_by = serializers.StringRelatedField()
    exif = ExifMetadataSerializer(read_only=True)
    faces = FaceDetectionSerializer(many=True, read_only=True)
    forensic_analysis = ForensicAnalysisSerializer(read_only=True)
    reverse_matches = ReverseImageMatchSerializer(many=True, read_only=True)
    fact_checks = FactCheckReferenceSerializer(many=True, read_only=True)
    report = DetectionReportSerializer(read_only=True)
    tags = TaggingResultSerializer(many=True, read_only=True)

    class Meta:
        model = ImageSubmission
        fields = [
            "id",
            "image",
            "original_filename",
            "sha256_hash",
            "file_size_bytes",
            "mime_type",
            "status",
            "submitted_by",
            "submitted_at",
            "completed_at",
            "exif",
            "faces",
            "forensic_analysis",
            "reverse_matches",
            "fact_checks",
            "report",
        ]   
        
class ReverseImageMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReverseImageMatch
        fields = ['id', 'page_url', 'page_title', 'image_url', 'domain', 'match_type', 'similarity_score']

class FactCheckReferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FactCheckReference
        fields = ['id', 'claim_text', 'claimant', 'publisher_name', 'publisher_url', 'rating', 'review_date']

class DetectionReportDetailSerializer(serializers.ModelSerializer):
    reverse_matches = ReverseImageMatchSerializer(many=True, read_only=True)
    fact_check_references = FactCheckReferenceSerializer(many=True, read_only=True)

    class Meta:
        model = DetectionReport
        fields = [
            'id', 'file_hash', 'status', 'authentic_probability',
            'deepfake_probability', 'legal_provisions', 'reverse_matches',
            'fact_check_references', 'created_at'
        ]
'''
from rest_framework import serializers
from .models import ImageSubmission, ReverseImageMatch, FactCheckReference


class ReverseImageMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReverseImageMatch
        fields = [
            'id',
            'page_url',
            'domain',
            'similarity_score',
            'match_type',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class FactCheckReferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FactCheckReference
        fields = [
            'id',
            'claim_text',
            'publisher_name',
            'publisher_url',
            'rating',
            'review_date',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class ImageSubmissionSerializer(serializers.ModelSerializer):
    reverse_matches = ReverseImageMatchSerializer(many=True, read_only=True)
    fact_check_references = FactCheckReferenceSerializer(many=True, read_only=True)

    class Meta:
        model = ImageSubmission
        fields = [
            'id',
            'sha256_hash',
            'original_filename',
            'file_size_bytes',
            'mime_type',
            'submitted_at',  # Updated from created_at
            'reverse_matches',
            'fact_check_references',
        ]
        read_only_fields = ['id', 'submitted_at', 'reverse_matches', 'fact_check_references']