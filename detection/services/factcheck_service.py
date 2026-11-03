import io
import os
import logging
import requests
from google.cloud import vision
import serpapi
from PIL import Image
from django.conf import settings

logger = logging.getLogger(__name__)
class FactCheckService:
    @staticmethod
    def _prepare_compressed_image(image_path: str, max_size_kb: int = 400) -> io.BytesIO:
        """Compresses image to stay under SerpApi's 500KB upload limit."""
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
        Uses Google Cloud Vision API Web Detection to find web matches.
        Falls back to SerpApi (Google Lens) if GCP credentials/billing fails.
        Returns an empty list if both fail.
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found for reverse search: {image_path}")
            return []

        results = []

        # --- Stage 1: Try Primary (Google Cloud Vision API) ---
        has_gcp_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or getattr(settings, "GOOGLE_VISION_KEY", None)
        
        if has_gcp_creds:
            try:
                client = vision.ImageAnnotatorClient()
                with open(image_path, "rb") as image_file:
                    content = image_file.read()

                image = vision.Image(content=content)
                response = client.web_detection(image=image)

                if not response.error.message:
                    web_detection = response.web_detection

                    if web_detection.full_matching_images:
                        for img in web_detection.full_matching_images[:5]:
                            results.append({
                                "page_url": img.url,
                                "image_url": img.url,
                                "domain": img.url.split("/")[2] if "//" in img.url else "",
                                "match_type": "EXACT",
                                "similarity_score": 1.0,
                            })

                    if web_detection.visually_similar_images:
                        for img in web_detection.visually_similar_images[:5]:
                            results.append({
                                "page_url": img.url,
                                "image_url": img.url,
                                "domain": img.url.split("/")[2] if "//" in img.url else "",
                                "match_type": "SIMILAR",
                                "similarity_score": 0.75,
                            })

                    if results:
                        return results

                logger.warning(f"GCP Vision API unavailable/error ({getattr(response.error, 'message', 'No matches')}). Falling through to SerpApi...")
            except Exception as e:
                logger.warning(f"GCP Vision execution failed ({e}). Falling through to SerpApi...")

        # --- Stage 2: Try Secondary (SerpApi / Google Lens) ---
        serpapi_key = getattr(settings, "SERPAPI_API_KEY", "") or os.getenv("SERPAPI_API_KEY", "")

        if serpapi_key:
            try:
                logger.info("Executing reverse image search via SerpApi...")
                
                # Compress buffer under 500KB limit
                compressed_buffer = FactCheckService._prepare_compressed_image(image_path, max_size_kb=400)

                upload_url = "https://serpapi.com/image"
                upload_resp = requests.post(
                    upload_url,
                    files={"image": ("upload.jpg", compressed_buffer, "image/jpeg")},
                    data={"api_key": serpapi_key}
                )

                upload_data = upload_resp.json()
                image_id = upload_data.get("image_id")

                if image_id:
                    client = serpapi.Client(api_key=serpapi_key)
                    params = {"engine": "google_lens", "image_id": image_id}
                    search_response = client.search(params)

                    visual_matches = search_response.get("visual_matches", [])
                    for match in visual_matches:
                        match_url = match.get("link", "")
                        if match_url:
                            results.append({
                                "page_url": match_url,
                                "image_url": match.get("thumbnail", match_url),
                                "domain": match_url.split("/")[2] if "//" in match_url else "",
                                "match_type": "SERPAPI_MATCH",
                                "similarity_score": 0.85,
                            })

                    if results:
                        return results

            except Exception as exc:
                logger.error(f"SerpApi reverse image search failed: {exc}")

        # --- Stage 3: Graceful Empty Fallback ---
        logger.info("No live reverse image matches found. Returning empty list.")
        return []

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
            "query": query_text if query_text.strip() else "image manipulation",
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

            if not references:
                logger.info("No live fact-check claims found. Returning mock claims.")
                return FactCheckService._get_mock_fact_checks()

        except Exception as e:
            logger.error(f"Fact Check Tools API query error: {str(e)}. Falling back to mock claims.")
            return FactCheckService._get_mock_fact_checks()

        return references