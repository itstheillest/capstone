import hashlib
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from .models import ImageSubmission
from .serializers import ImageSubmissionSerializer
from .tasks import process_submission_task


class ImageSubmissionViewSet(viewsets.ModelViewSet):
    queryset = ImageSubmission.objects.all().order_by("-submitted_at")
    serializer_class = ImageSubmissionSerializer
    parser_classes = [MultiPartParser, FormParser]

    def create(self, request, *args, **kwargs):
        uploaded_file = request.FILES.get("image")
        if not uploaded_file:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)

        # Generate SHA-256 hash from file content
        hasher = hashlib.sha256()
        for chunk in uploaded_file.chunks():
            hasher.update(chunk)
        sha256_hash = hasher.hexdigest()

        # Check for existing duplicate submission
        existing = ImageSubmission.objects.filter(sha256_hash=sha256_hash).first()
        if existing:
            serializer = self.get_serializer(existing)
            return Response(
                {"message": "Image already analyzed.", "data": serializer.data},
                status=status.HTTP_200_OK
            )

        # Save submission record
        submission = ImageSubmission.objects.create(
            image=uploaded_file,
            sha256_hash=sha256_hash,
            original_filename=uploaded_file.name,
            file_size_bytes=uploaded_file.size,
            mime_type=uploaded_file.content_type or "image/png",
            status=ImageSubmission.Status.PENDING,
        )

        # Dispatch async processing task
        actor_id = str(request.user.id) if request.user.is_authenticated else None
        process_submission_task.delay(str(submission.id), actor_id=actor_id)

        serializer = self.get_serializer(submission)
        return Response(serializer.data, status=status.HTTP_201_CREATED)