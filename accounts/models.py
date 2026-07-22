import uuid

from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Custom user model. Extends Django's built-in auth user so PixifAI can
    attach agency/role context without touching auth internals.
    """

    class Role(models.TextChoices):
        ANALYST = "analyst", "Analyst"
        SUPERVISOR = "supervisor", "Supervisor"
        ADMIN = "admin", "Administrator"
        VIEWER = "viewer", "Viewer (read-only)"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    role = models.CharField(max_length=20, choices=Role.choices, default=Role.ANALYST)
    agency = models.CharField(
        max_length=150,
        blank=True,
        help_text="Office/agency the user belongs to, e.g. CIO - Region IV-A",
    )
    is_verified_reviewer = models.BooleanField(
        default=False,
        help_text="Whether this user is authorized to finalize/override AI verdicts.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"
