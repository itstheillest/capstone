import hashlib
import io
import tempfile
import os
from unittest.mock import patch

from PIL import Image
from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from audit.models import AuditLog
from detection.models import (
    DetectionReport,
    ExifMetadata,
    ForensicAnalysis,
    ImageSubmission,
    TaggingResult,
)
from detection.services import face_service

User = get_user_model()


class ImageSubmissionAPITests(APITestCase):
    def setUp(self):
        call_command("seed_legal_provisions", stdout=io.StringIO())
        self.user = User.objects.create_user(
            username="testuser",
            password="testpassword123",
        )
        self.client.force_authenticate(user=self.user)

        image_buffer = io.BytesIO()
        image = Image.new("RGB", (100, 100), color="blue")
        image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)

        self.test_image = SimpleUploadedFile(
            name="test_image.jpg",
            content=image_buffer.read(),
            content_type="image/jpeg",
        )

    @patch("detection.services.factcheck_service.FactCheckService._query_google_news_rss", return_value=[])
    def test_upload_image_success(self, _mock_rss):
        response = self.client.post("/api/submissions/", {"image": self.test_image}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(ImageSubmission.objects.count(), 1)
        self.assertIn("id", response.data)

    def test_get_submission_detail(self):
        submission = ImageSubmission.objects.create(
            submitted_by=self.user,
            original_filename="test.jpg",
            sha256_hash=hashlib.sha256(b"test-detail").hexdigest(),
            file_size_bytes=1024,
            mime_type="image/jpeg",
        )
        response = self.client.get(f"/api/submissions/{submission.id}/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], str(submission.id))
        self.assertIsNone(response.data["exif"])
        self.assertIsNone(response.data["forensic_result"])
        self.assertIsNone(response.data["report"])
        self.assertEqual(response.data["faces"], [])
        self.assertEqual(response.data["tags"], [])


class PipelineIntegrationTests(APITestCase):
    def setUp(self):
        call_command("seed_legal_provisions", stdout=io.StringIO())
        self.user = User.objects.create_user(
            username="pipelinetester",
            password="testpassword123",
        )
        self.client.force_authenticate(user=self.user)

        image_buffer = io.BytesIO()
        image = Image.new("RGB", (800, 800), color="red")
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        self.test_image = SimpleUploadedFile(
            name="pipeline_test.png",
            content=image_buffer.read(),
            content_type="image/png",
        )

    def test_ycbcr_fallback_rejects_solid_image(self):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            Image.new("RGB", (128, 128), color=(210, 160, 140)).save(path, format="PNG")
            self.assertEqual(face_service._detect_faces_ycbcr(path), [])
        finally:
            if os.path.exists(path):
                os.remove(path)

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True, CELERY_TASK_EAGER_PROPAGATES=True)
    @patch("detection.services.factcheck_service.FactCheckService._query_google_news_rss", return_value=[])
    def test_full_pipeline_execution(self, _mock_rss):
        response = self.client.post("/api/submissions/", {"image": self.test_image}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        submission_id = response.data["id"]

        submission = ImageSubmission.objects.get(id=submission_id)
        self.assertEqual(submission.status, ImageSubmission.Status.COMPLETED)
        self.assertEqual(submission.submitted_by, self.user)

        self.assertTrue(ExifMetadata.objects.filter(submission=submission).exists())
        self.assertTrue(ForensicAnalysis.objects.filter(submission=submission).exists())
        self.assertTrue(TaggingResult.objects.filter(submission=submission).exists())
        self.assertTrue(DetectionReport.objects.filter(submission=submission).exists())
        self.assertTrue(AuditLog.objects.filter(submission=submission).exists())

        detail = self.client.get(f"/api/submissions/{submission_id}/")
        self.assertEqual(detail.status_code, status.HTTP_200_OK)
        for key in (
            "exif",
            "faces",
            "forensic_result",
            "tags",
            "report",
            "reverse_matches",
            "fact_check_references",
        ):
            self.assertIn(key, detail.data)
        self.assertIsNotNone(detail.data["report"])
        self.assertIn("legal_mappings", detail.data["report"])
        self.assertGreater(len(detail.data["report"]["legal_mappings"]), 0)
        self.assertEqual(detail.data["reverse_matches"], [])
        self.assertEqual(detail.data["fact_check_references"], [])
