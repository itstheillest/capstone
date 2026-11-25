"""
Web detection service using Google Cloud Vision API with SerpApi Google Lens fallback.
"""

import io
import logging
import os
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
        self._gcp_client = None
        self._gcp_init_attempted = False

    def _get_gcp_client(self):
        if self._gcp_init_attempted:
            return self._gcp_client

        self._gcp_init_attempted = True
        if not _has_gcp_vision:
            return None

        creds_path = (
            getattr(settings, "GOOGLE_APPLICATION_CREDENTIALS", "")
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        ).strip()

        try:
            if creds_path and os.path.exists(creds_path):
                credentials = service_account.Credentials.from_service_account_file(creds_path)
                self._gcp_client = vision.ImageAnnotatorClient(credentials=credentials)
        except Exception as exc:
            logger.warning("Could not initialize GCP Vision client: %s", exc)
            self._gcp_client = None

        return self._gcp_client

    def perform_web_detection(self, image_path: str = None, image_url: str = None) -> dict:
        serpapi_key = (getattr(settings, "SERPAPI_API_KEY", "") or os.getenv("SERPAPI_API_KEY", "")).strip()

        # --- Stage 1: Google Cloud Vision ---
        gcp_client = self._get_gcp_client()
        if gcp_client and image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as image_file:
                    content = image_file.read()

                image = vision.Image(content=content)
                response = gcp_client.web_detection(image=image)

                if not response.error.message:
                    return self._parse_gcp_response(response.web_detection)

                logger.warning("GCP Vision error (%s). Falling through to SerpApi...", response.error.message)
            except Exception as exc:
                logger.warning("GCP Vision failed (%s). Falling through to SerpApi...", exc)

        # --- Stage 2: SerpApi Google Lens with <400KB JPEG Compression ---
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
        with Image.open(image_path) as src:
            img = src.convert("RGB")
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
            response = client.search({"engine": "google_lens", "url": image_url})
        elif image_path and os.path.exists(image_path):
            compressed_buffer = self.prepare_compressed_image(image_path, max_size_kb=400)
            upload_resp = requests.post(
                "https://serpapi.com/image",
                files={"image": ("upload.jpg", compressed_buffer, "image/jpeg")},
                data={"api_key": serpapi_key},
                timeout=30,
            )
            upload_resp.raise_for_status()
            upload_data = upload_resp.json()
            image_id = upload_data.get("image_id")

            if not image_id:
                raise ValueError(f"SerpApi upload error: {upload_data}")

            client = serpapi.Client(api_key=serpapi_key)
            response = client.search({"engine": "google_lens", "image_id": image_id})
        else:
            raise ValueError(f"Invalid parameters: path={image_path}, url={image_url}")

        visual_matches = response.get("visual_matches", []) if hasattr(response, "get") else []
        for match in visual_matches:
            match_url = match.get("link")
            title = match.get("title", "")
            thumbnail = match.get("thumbnail", "")
            if match_url:
                results["pages_with_matching_images"].append({
                    "url": match_url,
                    "image_url": thumbnail,
                    "page_title": title,
                })
                if thumbnail:
                    results["full_matching_images"].append({"url": thumbnail})

        return results

    def _parse_gcp_response(self, annotations) -> dict:
        results = self._empty_payload()
        if not annotations:
            return results

        for page in getattr(annotations, "pages_with_matching_images", []) or []:
            matched_img = ""
            if getattr(page, "full_matching_images", None):
                matched_img = page.full_matching_images[0].url
            elif getattr(page, "partial_matching_images", None):
                matched_img = page.partial_matching_images[0].url

            results["pages_with_matching_images"].append({
                "url": page.url,
                "image_url": matched_img,
                "page_title": getattr(page, "page_title", ""),
            })

        for img in getattr(annotations, "full_matching_images", []) or []:
            results["full_matching_images"].append({"url": img.url})

        for img in getattr(annotations, "partial_matching_images", []) or []:
            results["partial_matching_images"].append({"url": img.url})

        for img in getattr(annotations, "visually_similar_images", []) or []:
            results["visually_similar_images"].append({"url": img.url})

        for label in getattr(annotations, "best_guess_labels", []) or []:
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
