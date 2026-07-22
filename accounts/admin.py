from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ("username", "email", "role", "agency", "is_verified_reviewer", "is_staff")
    list_filter = ("role", "agency", "is_verified_reviewer")
    fieldsets = UserAdmin.fieldsets + (
        ("PixifAI profile", {"fields": ("role", "agency", "is_verified_reviewer")}),
    )
