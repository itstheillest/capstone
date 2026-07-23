import os
import logging
import requests
from google.cloud import vision
from django.conf import settings

logger = logging.getLogger(__name__)

class FactCheckService:
    @staticmethod
    def perform_reverse_image_search(image_path: str) -> list[dict]:
        """
        Uses Google Cloud Vision API Web Detection to find exact, partial, and similar web images.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found for reverse search: {image_path}")
            return []

        # Graceful fallback if credentials are not configured
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and not getattr(settings, "GOOGLE_VISION_KEY", None):
            logger.warning("Google Cloud Vision credentials missing. Skipping Web Detection.")
            return []

        results = []
        try:
            client = vision.ImageAnnotatorClient()
            with open(image_path, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = client.web_detection(image=image)
            web_detection = response.web_detection

            # Process Full Matching Images
            if web_detection.full_matching_images:
                for img in web_detection.full_matching_images[:5]:
                    results.append({
                        "page_url": img.url,
                        "image_url": img.url,
                        "domain": img.url.split("/")[2] if "//" in img.url else "",
                        "match_type": "EXACT",
                        "similarity_score": 1.0,
                    })

            # Process Visually Similar Images
            if web_detection.visually_similar_images:
                for img in web_detection.visually_similar_images[:5]:
                    results.append({
                        "page_url": img.url,
                        "image_url": img.url,
                        "domain": img.url.split("/")[2] if "//" in img.url else "",
                        "match_type": "SIMILAR",
                        "similarity_score": 0.75,
                    })

        except Exception as e:
            logger.error(f"Google Vision API error: {str(e)}")

        return results

    @staticmethod
    def query_fact_check_tools(query_text: str) -> list[dict]:
        """
        Queries the Google Fact Check Tools API for published claim reviews.
        API Docs: https://developers.google.com/fact-check/tools/api
        """
        api_key = getattr(settings, "GOOGLE_FACT_CHECK_API_KEY", os.environ.get("GOOGLE_FACT_CHECK_API_KEY"))
        if not api_key or not query_text.strip():
            return []

        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query_text,
            "key": api_key,
            "languageCode": "en",
        }

        references = []
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                claims = data.get("claims", [])
                for claim in claims[:5]:
                    claim_text = claim.get("text", "")
                    claimant = claim.get("claimant", "")
                    reviews = claim.get("claimReview", [])

                    for review in reviews:
                        publisher = review.get("publisher", {}).get("name", "Unknown")
                        pub_url = review.get("url", "")
                        rating = review.get("textualRating", "Unrated")

                        references.append({
                            "claim_text": claim_text,
                            "claimant": claimant,
                            "publisher_name": publisher,
                            "publisher_url": pub_url,
                            "rating": rating,
                        })
        except Exception as e:
            logger.error(f"Fact Check Tools API query error: {str(e)}")

        return references