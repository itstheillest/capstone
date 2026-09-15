from rest_framework import serializers

from .models import LegalMapping, LegalProvision


class LegalProvisionSerializer(serializers.ModelSerializer):
    class Meta:
        model = LegalProvision
        fields = ["id", "law_name", "law_number", "section", "description", "status", "jurisdiction"]


class LegalMappingSerializer(serializers.ModelSerializer):
    provision = LegalProvisionSerializer(read_only=True)

    class Meta:
        model = LegalMapping
        fields = ["id", "provision", "trigger_condition", "notes", "flagged_at"]