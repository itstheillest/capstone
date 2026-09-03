import os
import logging
from google.cloud import vision
from google.oauth2 import service_account

logger = logging.getLogger(__name__)


class VisionService:
    def __init__(self):
        self.creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "pixifai-4ea9b6136cf4.json")
        self.client = self._init_client()

    def _init_client(self):
        """
        Dynamically load service account credentials if the file exists;
        otherwise fall back to standard Google environment initialization.
        """
        try:
            if os.path.exists(self.creds_path):
                credentials = service_account.Credentials.from_service_account_file(self.creds_path)
                return vision.ImageAnnotatorClient(credentials=credentials)
            logger.warning(f"Google credentials file not found at '{self.creds_path}'. Falling back to default auth.")
            return vision.ImageAnnotatorClient()
        except Exception as exc:
            logger.error(f"Failed to initialize Google Vision client: {exc}")
            return None

    def perform_web_detection(self, image_path: str) -> dict:
        """
        Performs web detection on a local image file using the Google Cloud Vision API.
        Extracts matching pages, domains, and full/partial image matches to feed into the database.
        """
        results = {
            "pages_with_matching_images": [],
            "full_matching_images": [],
            "partial_matching_images": [],
            "visually_similar_images": [],
            "best_guess_labels": []
        }

        if not self.client:
            logger.error("Vision API client is not initialized. Skipping web detection.")
            return results

        try:
            with open(image_path, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = self.client.web_detection(image=image)

            if response.error.message:
                logger.error(f"Google Vision API response error: {response.error.message}")
                return results

            annotations = response.web_detection

            if annotations:
                # Extract pages with matching images
                if annotations.pages_with_matching_images:
                    for page in annotations.pages_with_matching_images:
                        results["pages_with_matching_images"].append({
                            "url": page.url,
                            "page_title": page.page_title,
                        })

                # Extract full matches
                if annotations.full_matching_images:
                    for image_match in annotations.full_matching_images:
                        results["full_matching_images"].append({
                            "url": image_match.url
                        })

                # Extract partial matches
                if annotations.partial_matching_images:
                    for image_match in annotations.partial_matching_images:
                        results["partial_matching_images"].append({
                            "url": image_match.url
                        })

                # Extract visually similar images
                if annotations.visually_similar_images:
                    for image_match in annotations.visually_similar_images:
                        results["visually_similar_images"].append({
                            "url": image_match.url
                        })

                # Extract best guess labels / search keywords
                if annotations.best_guess_labels:
                    for label in annotations.best_guess_labels:
                        results["best_guess_labels"].append(label.label)

        except Exception as exc:
            logger.exception(f"Error performing web detection on {image_path}: {exc}")

        return results


# Export singleton instance for the pipeline
vision_service = VisionService()