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
    def query_fact_check_tools(query_text: str) -> dict:
        api_key = (
            getattr(settings, "FACT_CHECK_API_KEY", None)
            or os.environ.get("FACT_CHECK_API_KEY")
        )

        if not query_text or not query_text.strip():
            return {"status": "not_searched", "references": []}

        if not api_key:
            logger.warning("Fact-check lookup unavailable: API key is not configured.")
            return {"status": "unavailable", "references": []}

        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query_text.strip(),
            "key": api_key,
            "languageCode": "en",
            "pageSize": 5,
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                # Don't log the request URL: it contains the API key.
                logger.warning(
                    "Google Fact Check API returned HTTP %s.",
                    response.status_code,
                )
                return {"status": "unavailable", "references": []}

            references = []
            data = response.json()

            for claim in data.get("claims", []):
                for review in claim.get("claimReview", []) or []:
                    publisher_url = review.get("url", "")
                    if not publisher_url.startswith(("https://", "http://")):
                        continue

                    references.append({
                        "claim_text": claim.get("text", ""),
                        "claimant": claim.get("claimant", ""),
                        "publisher_name": (
                            review.get("publisher", {}).get("name", "Unknown")
                        ),
                        "publisher_url": publisher_url,
                        "rating": review.get("textualRating", "Unrated"),
                    })

            return {
                "status": "found" if references else "no_results",
                "references": references[:5],
            }

        except (requests.RequestException, ValueError, TypeError) as exc:
            # Log the exception type only; request errors can contain the API URL.
            logger.warning(
                "Google Fact Check lookup failed (%s).",
                type(exc).__name__,
            )
            return {"status": "unavailable", "references": []}
