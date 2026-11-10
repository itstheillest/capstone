"""
External Verification: Reverse Image Search (via vision_service) and
Google Fact Check Tools API integration.
"""

import os
import logging
import requests
from django.conf import settings
from detection.services.vision_service import vision_service, VisionService

logger = logging.getLogger(__name__)


class FactCheckService:
    @staticmethod
    def _prepare_compressed_image(image_path: str, max_size_kb: int = 400):
        return VisionService.prepare_compressed_image(image_path, max_size_kb=max_size_kb)

    @staticmethod
    def perform_reverse_image_search(image_path: str) -> list[dict]:
        """
        Delegates web detection to vision_service (GCP Vision -> SerpApi -> Empty Fallback)
        and formats matches for ReverseImageMatch model persistence.
        """
        if not os.path.exists(image_path):
            logger.error("Image not found for reverse search: %s", image_path)
            return []

        payload = vision_service.perform_web_detection(image_path=image_path)
        results = []

        for page in payload.get("pages_with_matching_images", [])[:5]:
            url = page.get("url", "")
            if url:
                results.append({
                    "page_url": url,
                    "image_url": page.get("image_url", url),
                    "domain": url.split("/")[2] if "//" in url else "",
                    "match_type": "EXACT",
                    "similarity_score": 0.95,
                })

        if not results:
            for img in payload.get("full_matching_images", [])[:5]:
                url = img.get("url", "")
                if url:
                    results.append({
                        "page_url": url,
                        "image_url": url,
                        "domain": url.split("/")[2] if "//" in url else "",
                        "match_type": "EXACT",
                        "similarity_score": 1.0,
                    })

        for img in payload.get("visually_similar_images", [])[:5]:
            url = img.get("url", "")
            if url:
                results.append({
                    "page_url": url,
                    "image_url": url,
                    "domain": url.split("/")[2] if "//" in url else "",
                    "match_type": "SIMILAR",
                    "similarity_score": 0.75,
                })

        return results

    @staticmethod
    def query_fact_check_tools(query_text: str) -> list[dict]:
        """
        Queries the Google Fact Check Tools API for published claim reviews.
        Returns only claim reviews returned by Google's API. Missing keys,
        empty results, and API failures produce no references rather than
        fabricated claims or links.
        """
        api_key = getattr(settings, "FACT_CHECK_API_KEY", os.environ.get("FACT_CHECK_API_KEY"))
        if not api_key:
            logger.warning("Google Fact Check API key missing; skipping fact-check lookup.")
            return []

        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query_text if query_text and query_text.strip() else "image manipulation",
            "key": api_key,
            "languageCode": "en",
        }

        references = []
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for claim in data.get("claims", [])[:5]:
                    claim_text = claim.get("text", "")
                    claimant = claim.get("claimant", "")
                    for review in claim.get("claimReview", []):
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

            if not references:
                logger.info("Google Fact Check API returned no matching claim reviews.")

        except Exception as exc:
            logger.error("Fact Check Tools API query error (%s).", exc)
            return []

        return references
