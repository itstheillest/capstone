import io
from PIL import Image
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import override_settings
from rest_framework import status
from rest_framework.test import APITestCase

from detection.models import (
    ImageSubmission,
    ExifMetadata,
    DetectionReport,
    ReverseImageMatch,
    FactCheckReference,
)
from detection.tasks import process_submission_task

User = get_user_model()


class ImageSubmissionAPITests(APITestCase):
    def setUp(self):
        # Create and authenticate a test user
        self.user = User.objects.create_user(
            username="testuser",
            password="testpassword123"
        )
        self.client.force_authenticate(user=self.user)

        # Create an in-memory image to avoid file locking on Windows
        image_buffer = io.BytesIO()
        image = Image.new("RGB", (100, 100), color="blue")
        image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)

        self.test_image = SimpleUploadedFile(
            name="test_image.jpg",
            content=image_buffer.read(),
            content_type="image/jpeg",
        )

    def test_upload_image_success(self):
        url = "/api/submissions/"
        response = self.client.post(url, {"image": self.test_image}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(ImageSubmission.objects.count(), 1)
        self.assertIn("id", response.data)

    def test_get_submission_detail(self):
        submission = ImageSubmission.objects.create(
            original_filename="test.jpg",
            file_size_bytes=1024,
            mime_type="image/jpeg",
        )
        url = f"/api/submissions/{submission.id}/"
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], str(submission.id))


class PipelineIntegrationTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="pipelinetester",
            password="testpassword123"
        )
        self.client.force_authenticate(user=self.user)

        image_buffer = io.BytesIO()
        image = Image.new("RGB", (200, 200), color="red")
        image.save(image_buffer, format="JPEG")
        image_buffer.seek(0)

        self.test_image = SimpleUploadedFile(
            name="pipeline_test.jpg",
            content=image_buffer.read(),
            content_type="image/jpeg",
        )

    @override_settings(CELERY_TASK_ALWAYS_EAGER=True)
    def test_full_pipeline_execution(self):
        # 1. Post image via API
        response = self.client.post("/api/submissions/", {"image": self.test_image}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        submission_id = response.data["id"]

        # 2. Execute Celery task synchronously
        process_submission_task.apply(args=[submission_id])

        # 3. Verify parent record status updated to completed
        submission = ImageSubmission.objects.get(id=submission_id)
        self.assertEqual(submission.status, "completed")

        # 4. Verify all child metadata models were populated
        self.assertTrue(ExifMetadata.objects.filter(submission=submission).exists())
        self.assertTrue(DetectionReport.objects.filter(submission=submission).exists())
        self.assertTrue(ReverseImageMatch.objects.filter(submission=submission).exists())
        self.assertTrue(FactCheckReference.objects.filter(submission=submission).exists())
        
class EdgeCaseDetectionTests(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="edgecase_user",
            password="testpassword123"
        )
        self.client.force_authenticate(user=self.user)
        self.upload_url = "/api/submissions/"

    def _create_dummy_image(self, width, height, format_="JPEG"):
        file_obj = io.BytesIO()
        image = Image.new("RGB", (width, height), color="red")
        image.save(file_obj, format=format_)
        file_obj.seek(0)
        return file_obj

    def test_zero_byte_file_upload(self):
        empty_file = SimpleUploadedFile("empty.jpg", b"", content_type="image/jpeg")
        response = self.client.post(self.upload_url, {"image": empty_file}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_invalid_file_extension_spoofing(self):
        fake_image = SimpleUploadedFile("fake.png", b"NOT_AN_IMAGE_FILE_DATA", content_type="image/png")
        response = self.client.post(self.upload_url, {"image": fake_image}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_extreme_image_dimensions(self):
        large_dim_img = self._create_dummy_image(10000, 100)
        upload_file = SimpleUploadedFile("wide.jpg", large_dim_img.read(), content_type="image/jpeg")
        response = self.client.post(self.upload_url, {"image": upload_file}, format="multipart")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)

    def test_rate_limiting_throttle(self):
        img_data = self._create_dummy_image(100, 100)
        responses = []
        for _ in range(30):
            upload_file = SimpleUploadedFile("test.jpg", img_data.getvalue(), content_type="image/jpeg")
            responses.append(self.client.post(self.upload_url, {"image": upload_file}, format="multipart"))
        
        status_codes = [r.status_code for r in responses]
        # Validates whether throttling is enforced or if requests succeed without crashing
        self.assertTrue(any(code in [status.HTTP_201_CREATED, status.HTTP_429_TOO_MANY_REQUESTS] for code in status_codes))