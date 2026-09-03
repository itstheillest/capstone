'''
import hashlib

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from audit.models import AuditLog

from .models import ImageSubmission
from .serializers import (
    ImageSubmissionDetailSerializer,
    ImageSubmissionListSerializer,
    ImageSubmissionUploadSerializer,
)


class ImageSubmissionViewSet(viewsets.ModelViewSet):
    """
    /api/submissions/          GET (list), POST (upload)
    /api/submissions/{id}/     GET (full detail w/ nested pipeline results)
    /api/submissions/{id}/status/ GET (lightweight status polling)
    """

    permission_classes = [IsAuthenticated]
    http_method_names = ["get", "post", "head", "options"]  # no edit/delete via API

    def get_queryset(self):
        user = self.request.user
        qs = ImageSubmission.objects.select_related("submitted_by")
        if not user.is_staff:
            qs = qs.filter(submitted_by=user)
        return qs

    def get_serializer_class(self):
        if self.action == "create":
            return ImageSubmissionUploadSerializer
        if self.action == "list":
            return ImageSubmissionListSerializer
        return ImageSubmissionDetailSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        image_file = serializer.validated_data["image"]

        sha256 = hashlib.sha256()
        for chunk in image_file.chunks():
            sha256.update(chunk)
        file_hash = sha256.hexdigest()
        image_file.seek(0)

        existing = ImageSubmission.objects.filter(sha256_hash=file_hash).first()
        if existing:
            AuditLog.objects.create(
                submission=existing,
                actor=request.user,
                action_type=AuditLog.ActionType.UPLOAD,
                details={"note": "Duplicate upload detected; returned existing submission."},
                ip_address=self._client_ip(request),
            )
            out = ImageSubmissionDetailSerializer(existing, context={"request": request})
            return Response(out.data, status=status.HTTP_200_OK)

        submission = ImageSubmission.objects.create(
            submitted_by=request.user,
            image=image_file,
            original_filename=image_file.name,
            sha256_hash=file_hash,
            file_size_bytes=image_file.size,
            mime_type=getattr(image_file, "content_type", "") or "",
            status=ImageSubmission.Status.PENDING,
        )

        AuditLog.objects.create(
            submission=submission,
            actor=request.user,
            action_type=AuditLog.ActionType.UPLOAD,
            details={"original_filename": submission.original_filename},
            ip_address=self._client_ip(request),
        )

        # Kick off async processing: EXIF check, face detection, forensic analysis.
        from detection.tasks import process_submission_task

        process_submission_task.delay(str(submission.id), actor_id=str(request.user.id))
        
        # In production this task runs on a separate worker, so `submission`
        # stays accurate. But CELERY_TASK_ALWAYS_EAGER (dev/testing) runs it
        # synchronously right here, which updates the DB row without
        # updating this in-memory object — refresh so the response reflects
        # what actually happened either way.
        submission.refresh_from_db()

        out = ImageSubmissionDetailSerializer(submission, context={"request": request})
        return Response(out.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        """
        GET /api/submissions/{id}/status/
        Endpoint to allow frontend clients to lightweight-poll pipeline completion.
        """
        submission = self.get_object()
        return Response({
            "id": str(submission.id),
            "status": submission.status,
            "processed_at": submission.processed_at,
        }, status=status.HTTP_200_OK)

    @staticmethod
    def _client_ip(request):
        forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR")
'''
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import ImageSubmission
from .serializers import ImageSubmissionSerializer

class ImageSubmissionViewSet(viewsets.ModelViewSet):
    """
    API endpoint for managing image submissions and inspecting detection results.
    """
    queryset = ImageSubmission.objects.all().prefetch_related(
        'reverse_matches', 
        'fact_check_references'
    ).order_by('-submitted_at')  # Updated from -created_at
    
    serializer_class = ImageSubmissionSerializer

    @action(detail=True, methods=['post'])
    def trigger_analysis(self, request, pk=None):
        """
        Custom endpoint hook to trigger asynchronous analysis.
        Can be tested independently before hooking into your team's Celery pipeline.
        """
        submission = self.get_object()
        
        # Stub response for independent testing
        return Response(
            {
                "status": "queued",
                "message": f"Analysis initiated for submission {submission.id}.",
                "sha256_hash": submission.sha256_hash,
            },
            status=status.HTTP_202_ACCEPTED
        )