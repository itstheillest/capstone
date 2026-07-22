from django.contrib import admin

from .models import AuditLog


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ("timestamp", "action_type", "submission", "actor", "ip_address")
    list_filter = ("action_type",)
    readonly_fields = [f.name for f in AuditLog._meta.fields]

    def has_delete_permission(self, request, obj=None):
        # Audit trail must be append-only.
        return False

    def has_change_permission(self, request, obj=None):
        return False
