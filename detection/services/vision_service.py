"""
Web detection service using Google Cloud Vision API with SerpApi Google Lens fallback.
"""

import io
import os
import logging
import requests
from PIL import Image
from django.conf import settings

logger = logging.getLogger(__name__)

try:
    from google.cloud import vision
    from google.oauth2 import service_account
    _has_gcp_vision = True
except ImportError:
    vision = None
    service_account = None
    _has_gcp_vision = False

try:
    import serpapi
    _has_serpapi = True
except ImportError:
    serpapi = None
    _has_serpapi = False


class VisionService:
    def __init__(self):
        self.gcp_creds_path = getattr(
            settings,
            "GOOGLE_APPLICATION_CREDENTIALS",
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "pixifai-4ea9b6136cf4.json"),
        )
        self.gcp_client = self._init_gcp_client()

    def _init_gcp_client(self):
        if not _has_gcp_vision:
            return None
        try:
            if self.gcp_creds_path and os.path.exists(self.gcp_creds_path):
                credentials = service_account.Credentials.from_service_account_file(self.gcp_creds_path)
                return vision.ImageAnnotatorClient(credentials=credentials)
            return None
        except Exception as exc:
            logger.warning("Could not initialize GCP Vision client: %s", exc)
            return None

    def perform_web_detection(self, image_path: str = None, image_url: str = None) -> dict:
        serpapi_key = getattr(settings, "SERPAPI_API_KEY", "") or os.getenv("SERPAPI_API_KEY", "")

        # --- Stage 1: Google Cloud Vision ---
        if self.gcp_client and image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as image_file:
                    content = image_file.read()

                image = vision.Image(content=content)
                response = self.gcp_client.web_detection(image=image)

                if not response.error.message:
                    return self._parse_gcp_response(response.web_detection)

                logger.warning("GCP Vision error (%s). Falling through to SerpApi...", response.error.message)
            except Exception as exc:
                logger.warning("GCP Vision failed (%s). Falling through to SerpApi...", exc)

        # --- Stage 2: SerpApi with Image Compression ---
        if serpapi_key and _has_serpapi:
            try:
                logger.info("Executing web detection via SerpApi...")
                return self._run_serpapi_detection(serpapi_key, image_path, image_url)
            except Exception as exc:
                logger.warning("SerpApi execution failed: %s", exc)

        # --- Stage 3: Empty Payload ---
        return self._empty_payload()

    @staticmethod
    def prepare_compressed_image(image_path: str, max_size_kb: int = 400) -> io.BytesIO:
        """Resizes and compresses image in-memory under 400KB to prevent SerpApi 400 errors."""
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        max_dim = 1024
        if max(img.width, img.height) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        quality = 85
        buffer = io.BytesIO()

        while quality >= 20:
            buffer.seek(0)
            buffer.truncate()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            if buffer.tell() / 1024 <= max_size_kb:
                break
            quality -= 10

        buffer.seek(0)
        return buffer

    def _run_serpapi_detection(self, serpapi_key: str, image_path: str = None, image_url: str = None) -> dict:
        results = self._empty_payload()

        if image_url:
            client = serpapi.Client(api_key=serpapi_key)
            params = {"engine": "google_lens", "url": image_url}
            response = client.search(params)
        elif image_path and os.path.exists(image_path):
            compressed_buffer = self.prepare_compressed_image(image_path, max_size_kb=400)
            upload_resp = requests.post(
                "https://serpapi.com/image",
                files={"image": ("upload.jpg", compressed_buffer, "image/jpeg")},
                data={"api_key": serpapi_key},
                timeout=30,
            )
            upload_data = upload_resp.json()
            image_id = upload_data.get("image_id")

            if not image_id:
                raise ValueError(f"SerpApi upload error: {upload_data}")

            client = serpapi.Client(api_key=serpapi_key)
            params = {"engine": "google_lens", "image_id": image_id}
            response = client.search(params)
        else:
            raise ValueError(f"Invalid parameters: path={image_path}, url={image_url}")

        for match in response.get("visual_matches", []):
            match_url = match.get("link")
            title = match.get("title", "")
            thumbnail = match.get("thumbnail", match_url)
            if match_url:
                results["pages_with_matching_images"].append({
                    "url": match_url,
                    "image_url": thumbnail,
                    "page_title": title,
                })
                results["full_matching_images"].append({"url": thumbnail or match_url})

        return results

    def _parse_gcp_response(self, annotations) -> dict:
        results = self._empty_payload()
        if annotations:
            if annotations.pages_with_matching_images:
                for page in annotations.pages_with_matching_images:
                    results["pages_with_matching_images"].append({
                        "url": page.url,
                        "image_url": page.url,
                        "page_title": getattr(page, "page_title", ""),
                    })
            if annotations.full_matching_images:
                for img in annotations.full_matching_images:
                    results["full_matching_images"].append({"url": img.url})
            if annotations.visually_similar_images:
                for img in annotations.visually_similar_images:
                    results["visually_similar_images"].append({"url": img.url})
            if annotations.best_guess_labels:
                for label in annotations.best_guess_labels:
                    results["best_guess_labels"].append(label.label)
        return results

    @staticmethod
    def _empty_payload() -> dict:
        return {
            "pages_with_matching_images": [],
            "full_matching_images": [],
            "partial_matching_images": [],
            "visually_similar_images": [],
            "best_guess_labels": [],
        }


vision_service = VisionService()