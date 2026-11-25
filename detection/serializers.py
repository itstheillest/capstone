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


class ExifMetadataSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExifMetadata
        fields = [
            "camera_make",
            "camera_model",
            "software_tag",
            "capture_timestamp",
            "gps_latitude",
            "gps_longitude",
            "has_discrepancy",
            "discrepancy_notes",
            "raw_exif",
        ]


class FaceDetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceDetection
        fields = [
            "id",
            "face_index",
            "bounding_box",
            "detection_confidence",
            "deepface_emotion",
            "deepface_age_estimate",
        ]


class ForensicAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForensicAnalysis
        fields = [
            "id",
            "model_name",
            "model_version",
            "manipulation_score",
            "verdict",
            "anomaly_regions",
            "heatmap_image",
            "processing_time_ms",
            "analyzed_at",
        ]


class TaggingResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = TaggingResult
        fields = [
            "id",
            "tag_code",
            "tag_label",
            "rule_description",
            "confidence",
            "created_at",
        ]


class DetectionReportSerializer(serializers.ModelSerializer):
    legal_mappings = LegalMappingSerializer(many=True, read_only=True)

    class Meta:
        model = DetectionReport
        fields = [
            "id",
            "overall_verdict",
            "confidence_score",
            "summary_text",
            "review_status",
            "generated_at",
            "reviewed_at",
            "legal_mappings",
        ]


class ReverseImageMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReverseImageMatch
        fields = [
            "id",
            "page_url",
            "page_title",
            "image_url",
            "domain",
            "match_type",
            "similarity_score",
            "created_at",
        ]


class FactCheckReferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FactCheckReference
        fields = [
            "id",
            "claim_text",
            "claimant",
            "publisher_name",
            "publisher_url",
            "rating",
            "review_date",
            "created_at",
        ]


class ImageSubmissionSerializer(serializers.ModelSerializer):
    exif = ExifMetadataSerializer(read_only=True)
    faces = FaceDetectionSerializer(many=True, read_only=True)
    forensic_result = ForensicAnalysisSerializer(source="forensic_analysis", read_only=True)
    tags = TaggingResultSerializer(many=True, read_only=True)
    report = DetectionReportSerializer(read_only=True)
    reverse_matches = ReverseImageMatchSerializer(many=True, read_only=True)
    fact_check_references = FactCheckReferenceSerializer(many=True, read_only=True)

    class Meta:
        model = ImageSubmission
        fields = [
            "id",
            "image",
            "original_filename",
            "file_size_bytes",
            "mime_type",
            "sha256_hash",
            "status",
            "submitted_at",
            "completed_at",
            "exif",
            "faces",
            "forensic_result",
            "tags",
            "report",
            "reverse_matches",
            "fact_check_references",
        ]
        read_only_fields = [
            "id",
            "original_filename",
            "file_size_bytes",
            "mime_type",
            "sha256_hash",
            "status",
            "submitted_at",
            "completed_at",
        ]
