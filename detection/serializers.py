from rest_framework import serializers
from .models import (
    ImageSubmission,
    DetectionReport,
    ReverseImageMatch,
    FactCheckReference,
)


def _get_all_related_objects(instance):
    """Yields (accessor_name, related_object_or_qs, is_many) for all reverse relations."""
    for rel in instance._meta.related_objects:
        accessor = rel.get_accessor_name()
        is_many = rel.one_to_many or rel.many_to_many
        try:
            val = getattr(instance, accessor, None)
            if val is not None:
                yield accessor, val, is_many, rel.related_model
        except Exception:
            continue


def _find_single_by_field(instance, field_name):
    """Finds a 1-to-1 or FK related object that possesses `field_name`."""
    for _, val, is_many, model_cls in _get_all_related_objects(instance):
        model_fields = {f.name for f in model_cls._meta.get_fields()}
        if field_name in model_fields:
            if is_many and hasattr(val, "first"):
                obj = val.first()
                if obj is not None:
                    return obj
            elif not is_many:
                return val
    return None


def _find_many_by_field(instance, field_name):
    """Finds all reverse related rows whose model contains `field_name`."""
    for _, val, is_many, model_cls in _get_all_related_objects(instance):
        model_fields = {f.name for f in model_cls._meta.get_fields()}
        if field_name in model_fields and is_many and hasattr(val, "all"):
            return list(val.all())
    return []


class DetectionReportSerializer(serializers.ModelSerializer):
    legal_mappings = serializers.SerializerMethodField()

    class Meta:
        model = DetectionReport
        fields = [
            "id",
            "overall_verdict",
            "confidence_score",
            "summary_text",
            "review_status",
            "legal_mappings",
        ]

    def get_legal_mappings(self, obj):
        items = _find_many_by_field(obj, "provision")
        if not items and hasattr(obj, "submission") and obj.submission is not None:
            items = _find_many_by_field(obj.submission, "provision")

        results = []
        for m in items:
            p = getattr(m, "provision", None)
            results.append({
                "id": str(getattr(m, "id", "")),
                "trigger_tag_code": getattr(m, "trigger_tag_code", getattr(m, "tag_code", "")),
                "relevance_explanation": getattr(
                    m, "relevance_explanation", getattr(m, "explanation", getattr(m, "rationale", ""))
                ),
                "provision": {
                    "law_code": getattr(p, "law_code", getattr(p, "code", "")),
                    "title": getattr(p, "title", getattr(p, "name", "")),
                    "section": getattr(p, "section", ""),
                    "status": getattr(p, "status", "enacted"),
                    "summary": getattr(p, "summary", getattr(p, "description", "")),
                    "penalty_description": getattr(p, "penalty_description", ""),
                } if p else {},
            })
        return results


class ReverseImageMatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = ReverseImageMatch
        fields = ["page_url", "image_url", "domain", "match_type", "similarity_score"]


class FactCheckReferenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = FactCheckReference
        fields = ["claim_text", "claimant", "publisher_name", "publisher_url", "rating"]


class ImageSubmissionSerializer(serializers.ModelSerializer):
    exif = serializers.SerializerMethodField()
    faces = serializers.SerializerMethodField()
    forensic_result = serializers.SerializerMethodField()
    tags = serializers.SerializerMethodField()
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
            "sha256_hash",
            "status",
            "submitted_at",
            "completed_at",
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
        return {
            "model_name": getattr(f_obj, "model_name", "PixifAI Spectral + ELA Heuristic Engine"),
            "model_version": getattr(f_obj, "model_version", "v1.1"),
            "manipulation_score": getattr(f_obj, "manipulation_score", 0.0),
            "verdict": getattr(f_obj, "verdict", "authentic"),
            "anomaly_regions": getattr(f_obj, "anomaly_regions", []),
        }

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