from django.contrib import admin

from .models import LegalMapping, LegalProvision


@admin.register(LegalProvision)
class LegalProvisionAdmin(admin.ModelAdmin):
    list_display = ("law_number", "section", "law_name", "jurisdiction")
    search_fields = ("law_number", "law_name", "section")


@admin.register(LegalMapping)
class LegalMappingAdmin(admin.ModelAdmin):
    list_display = ("report", "provision", "trigger_condition", "flagged_at")
    list_filter = ("trigger_condition",)
