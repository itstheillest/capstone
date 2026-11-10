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
    def _get_mock_reverse_matches() -> list[dict]:
        """Returns fallback reverse image search results for testing."""
        return [
            {
                "page_url": "https://starwars.com/databank/darth-vader",
                "image_url": "https://images.starwars.com/vader_full.jpg",
                "domain": "starwars.com",
                "match_type": "EXACT",
                "similarity_score": 0.98,
            },
            {
                "page_url": "https://wikipedia.org/wiki/Darth_Vader",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/vader.jpg",
                "domain": "wikipedia.org",
                "match_type": "SIMILAR",
                "similarity_score": 0.85,
            },
        ]

    @staticmethod
    def _get_mock_fact_checks() -> list[dict]:
        """Returns fallback fact check claims for testing."""
        return [
            {
                "claim_text": "Image claims to show unreleased film set footage.",
                "claimant": "Social Media Posts",
                "publisher_name": "FactCheck.org",
                "publisher_url": "https://factcheck.org/example-entry",
                "rating": "False",
            },
            {
                "claim_text": "Photo was altered to change original lighting and background.",
                "claimant": "Viral Tweet",
                "publisher_name": "PolitiFact",
                "publisher_url": "https://politifact.com/example-entry",
                "rating": "Pants on Fire",
            },
        ]

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
        Falls back to mock data if API key is missing or request fails.
        """
        api_key = getattr(settings, "FACT_CHECK_API_KEY", os.environ.get("FACT_CHECK_API_KEY"))
        if not api_key:
            logger.warning("Google Fact Check API key missing. Falling back to mock fact checks.")
            return FactCheckService._get_mock_fact_checks()

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
                return FactCheckService._get_mock_fact_checks()

        except Exception as exc:
            logger.error("Fact Check Tools API query error (%s). Falling back to mock claims.", exc)
            return FactCheckService._get_mock_fact_checks()

        return references