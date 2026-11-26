import hashlib
from rest_framework import serializers, status, viewsets
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.decorators import action
from drf_spectacular.utils import extend_schema, inline_serializer
from django.db import transaction

from audit.models import AuditLog
from .models import ImageSubmission
from .serializers import ImageSubmissionSerializer
from .tasks import process_submission_task, retry_fact_check_task


class ImageSubmissionViewSet(viewsets.ModelViewSet):
    queryset = (
        ImageSubmission.objects
        .select_related("exif", "forensic_analysis", "report")
        .prefetch_related(
            "faces",
            "tags",
            "reverse_matches",
            "fact_check_references",
            "report__legal_mappings__provision",
        )
        .order_by("-submitted_at")
    )
    serializer_class = ImageSubmissionSerializer
    parser_classes = [MultiPartParser, FormParser]

    @extend_schema(
        request={
            "multipart/form-data": inline_serializer(
                name="ImageUploadRequest",
                fields={
                    "image": serializers.FileField(),
                },
            )
        }
    )

    def create(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get("image")
        if not uploaded_file:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)

        hasher = hashlib.sha256()
        for chunk in uploaded_file.chunks():
            hasher.update(chunk)
        sha256_hash = hasher.hexdigest()

        actor = request.user if request.user.is_authenticated else None
        actor_id = str(actor.id) if actor else None

        existing = self.get_queryset().filter(sha256_hash=sha256_hash).first()

        if existing:
            if existing.status == ImageSubmission.Status.FAILED:
                # The previous attempt may have deleted its temporary image.
                # Reattach the file the user just uploaded.
                existing.image.save(
                    uploaded_file.name,
                    uploaded_file,
                    save=False,
                )
                existing.status = ImageSubmission.Status.PENDING
                existing.save(update_fields=["image", "status"])

                process_submission_task.delay(
                    str(existing.id),
                    actor_id=actor_id,
                )

            serializer = self.get_serializer(existing)
            return Response(
                {"message": "Image already analyzed.", "data": serializer.data},
                status=status.HTTP_200_OK,
            )

        submission = ImageSubmission.objects.create(
            submitted_by=actor,
            image=uploaded_file,
            sha256_hash=sha256_hash,
            original_filename=uploaded_file.name,
            file_size_bytes=uploaded_file.size,
            mime_type=uploaded_file.content_type or "image/png",
            status=ImageSubmission.Status.PENDING,
        )

        AuditLog.objects.create(
            submission=submission,
            actor=actor,
            action_type=AuditLog.ActionType.UPLOAD,
            details={"filename": submission.original_filename, "size_bytes": submission.file_size_bytes},
            ip_address=request.META.get("REMOTE_ADDR"),
        )

        process_submission_task.delay(str(submission.id), actor_id=actor_id)
        submission.refresh_from_db()

        serializer = self.get_serializer(submission)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["post"], url_path="retry-fact-check")
    def retry_fact_check(self, request, pk=None):
        submission = self.get_object()

        if submission.status != ImageSubmission.Status.COMPLETED:
            return Response(
                {"detail": "Fact-check retry is available after analysis completes."},
                status=status.HTTP_409_CONFLICT,
            )

        if not submission.extracted_text.strip():
            return Response(
                {"detail": "This submission has no OCR text to search."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        with transaction.atomic():
            # Lock this submission so two clicks can't queue two retries.
            locked_submission = ImageSubmission.objects.select_for_update().get(
                pk=submission.pk
            )

            latest_log = locked_submission.audit_logs.filter(
                action_type=AuditLog.ActionType.FACT_CHECK
            ).order_by("-timestamp").first()
            latest_status = (latest_log.details or {}).get("fact_check_status")

            if latest_status == "pending":
                return Response(
                    {"detail": "A fact-check retry is already running."},
                    status=status.HTTP_409_CONFLICT,
                )

            if latest_status != "unavailable":
                return Response(
                    {
                        "detail": (
                            "Retry is available only when the latest "
                            "fact-check lookup is unavailable."
                        )
                    },
                    status=status.HTTP_409_CONFLICT,
                )

            AuditLog.objects.create(
                submission=submission,
                actor=request.user if request.user.is_authenticated else None,
                action_type=AuditLog.ActionType.FACT_CHECK,
                details={
                    "fact_check_status": "pending",
                    "fact_checks_found": 0,
                    "retry": True,
                },
            )

            transaction.on_commit(
                lambda: retry_fact_check_task.delay(str(submission.id))
            )

        return Response(
            {"status": "pending"},
            status=status.HTTP_202_ACCEPTED,
        )
