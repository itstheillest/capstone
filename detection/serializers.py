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
    fact_check_status = serializers.SerializerMethodField()
    experimental_classifier = serializers.SerializerMethodField()

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
            "fact_check_status",
            "ocr_status",
            "extracted_text",
            "ocr_confidence",
            "experimental_classifier",
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
            "ocr_status",
            "extracted_text",
            "ocr_confidence",
        ]

    def get_exif(self, obj):
        exif_obj = _find_single_by_field(obj, "has_discrepancy")
        if not exif_obj:
            return {
                "camera_make": "",
                "camera_model": "",
                "software_tag": "",
                "capture_timestamp": None,
                "gps_latitude": None,
                "gps_longitude": None,
                "has_discrepancy": True,
                "discrepancy_notes": "No EXIF metadata present — may indicate stripping, re-saving, or synthetic origin.",
            }
        return {
            "camera_make": getattr(exif_obj, "camera_make", ""),
            "camera_model": getattr(exif_obj, "camera_model", ""),
            "software_tag": getattr(exif_obj, "software_tag", ""),
            "capture_timestamp": getattr(exif_obj, "capture_timestamp", None),
            "gps_latitude": getattr(exif_obj, "gps_latitude", None),
            "gps_longitude": getattr(exif_obj, "gps_longitude", None),
            "has_discrepancy": getattr(exif_obj, "has_discrepancy", True),
            "discrepancy_notes": getattr(exif_obj, "discrepancy_notes", ""),
            "c2pa": (getattr(exif_obj, "raw_exif", {}) or {}).get("PixifAI:C2PA", {}),
        }

    def get_faces(self, obj):
        face_objs = _find_many_by_field(obj, "bounding_box")
        return [
            {
                "id": str(getattr(f, "id", "")),
                "bounding_box": getattr(f, "bounding_box", []),
                "detection_confidence": getattr(f, "detection_confidence", 0.0),
                "deepface_emotion": getattr(f, "deepface_emotion", ""),
                "deepface_age_estimate": getattr(f, "deepface_age_estimate", None),
            }
            for f in face_objs
        ]

    def get_forensic_result(self, obj):
        f_obj = _find_single_by_field(obj, "manipulation_score")
        if not f_obj:
            return None

        model_name = getattr(
            f_obj, "model_name", "PixifAI Spectral + ELA Heuristic Engine"
        )
        score = getattr(f_obj, "manipulation_score", 0.0)

        return {
            "model_name": model_name,
            "model_version": getattr(f_obj, "model_version", "v1.1"),
            "manipulation_score": score,
            "ai_probability": score if "AI-vs-Real" in model_name else None,
            "verdict": getattr(f_obj, "verdict", "authentic"),
            "anomaly_regions": getattr(f_obj, "anomaly_regions", []),
        }
        
    def get_experimental_classifier(self, obj):
        log = obj.audit_logs.filter(action_type="forensic_analysis").first()
        if not log:
            return None
        return (log.details or {}).get("teachable_machine")
    

    def get_tags(self, obj):
        tag_objs = _find_many_by_field(obj, "tag_code")
        return [
            {
                "tag_code": getattr(t, "tag_code", ""),
                "tag_label": getattr(t, "tag_label", ""),
                "confidence": getattr(t, "confidence", 0.0),
                "rule_description": getattr(t, "rule_description", ""),
            }
            for t in tag_objs
        ]
    
    def get_fact_check_status(self, obj):
        log = obj.audit_logs.filter(action_type="fact_check").first()
        if not log:
            return "pending"
        return (log.details or {}).get("fact_check_status", "unavailable")
