from PIL import Image
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
    image = serializers.ImageField(required=True)
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
    
    def validate_image(self, value):
        # 1. Reject zero-byte / empty files
        if value.size == 0:
            raise serializers.ValidationError("Empty or zero-byte files are not allowed.")

        # 2. Verify file header & integrity using Pillow
        try:
            img = Image.open(value)
            img.verify()  # Validates image format and header structure
        except Exception:
            raise serializers.ValidationError("Uploaded file is spoofed or not a valid image format.")

        # Reset file pointer after verify() reads the byte stream
        value.seek(0)
        return value

    def validate(self, attrs):
        image = attrs.get("image") or self.initial_data.get("image")
        
        if not image or getattr(image, "size", 0) == 0:
            raise serializers.ValidationError({"image": "Empty or zero-byte files are not allowed."})

        try:
            img = Image.open(image)
            img.verify()
            if hasattr(image, "seek"):
                image.seek(0)
        except Exception:
            raise serializers.ValidationError({"image": "Uploaded file is spoofed or not a valid image format."})

        return attrs
