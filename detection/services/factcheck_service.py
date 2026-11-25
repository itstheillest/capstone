"""
External Verification: Reverse Image Search (via vision_service) and
Google Fact Check Tools API with live keyless Google News RSS fallback.
"""

from datetime import date, datetime
from email.utils import parsedate_to_datetime
import logging
import time
import os
import re
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import requests
from django.conf import settings
from detection.services.vision_service import vision_service, VisionService

logger = logging.getLogger(__name__)

_QUERY_STOPWORDS = frozenset({
    "and", "the", "for", "with", "from", "that", "this", "no", "not",
    "detected", "requires", "human", "review", "likely", "authentic",
    "media", "none", "image",
})

_FACT_CHECK_SITES = (
    "site:verafiles.org OR site:rappler.com OR site:factcheck.afp.com "
    "OR site:politifact.com OR site:snopes.com OR site:factcheck.org"
)


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def _parse_review_date(raw_date: str | None) -> date | None:
    if not raw_date:
        return None
    text = str(raw_date).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except Exception:
        return None


def _sanitize_search_query(query_text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]{2,}", query_text or "")
    filtered = [t for t in tokens if t.lower() not in _QUERY_STOPWORDS]
    if not filtered:
        return "synthetic media deepfake"
    return " ".join(filtered[:5])


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
        if not image_path or not os.path.exists(image_path):
            logger.error("Image not found for reverse search: %s", image_path)
            return []

        payload = vision_service.perform_web_detection(image_path=image_path)
        results = []
        seen_urls = set()

        def _append_match(page_url: str, image_url: str, page_title: str, match_type: str, score: float):
            if not page_url or page_url in seen_urls:
                return
            seen_urls.add(page_url)
            results.append({
                "page_url": page_url,
                "page_title": (page_title or "")[:512],
                "image_url": image_url or "",
                "domain": _extract_domain(page_url)[:255],
                "match_type": match_type,
                "similarity_score": score,
            })

        for page in payload.get("pages_with_matching_images", [])[:5]:
            _append_match(
                page_url=page.get("url", ""),
                image_url=page.get("image_url", ""),
                page_title=page.get("page_title", ""),
                match_type="EXACT",
                score=0.95,
            )

        if not results:
            for img in payload.get("full_matching_images", [])[:5]:
                url = img.get("url", "")
                _append_match(
                    page_url=url,
                    image_url=url,
                    page_title="",
                    match_type="EXACT",
                    score=1.0,
                )

        for img in payload.get("partial_matching_images", [])[:5]:
            url = img.get("url", "")
            _append_match(
                page_url=url,
                image_url=url,
                page_title="",
                match_type="PARTIAL",
                score=0.85,
            )

        for img in payload.get("visually_similar_images", [])[:5]:
            url = img.get("url", "")
            _append_match(
                page_url=url,
                image_url=url,
                page_title="",
                match_type="SIMILAR",
                score=0.75,
            )

        return results

    @staticmethod
    def _query_google_news_rss(clean_query: str) -> list[dict]:
        """
        Live keyless fallback querying Google News RSS for verified fact-check articles.
        Assigns neutral 'Reference' rating so mapping_service does not falsely treat
        contextual articles as debunked claims.
        """
        rss_url = "https://news.google.com/rss/search"
        queries = [
            f'{clean_query} ("fact check" OR {_FACT_CHECK_SITES})',
            f'{clean_query} "fact check"',
        ]

        for rss_query in queries:
            try:
                resp = requests.get(
                    rss_url,
                    params={"q": rss_query, "hl": "en-PH", "gl": "PH", "ceid": "PH:en"},
                    headers={"User-Agent": "PixifAI/1.0"},
                    timeout=6,
                )
                if resp.status_code != 200 or not resp.content:
                    continue

                root = ET.fromstring(resp.content)
                items = root.findall("./channel/item")[:5]
                if not items:
                    continue

                references = []
                for item in items:
                    raw_title = (item.findtext("title") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    pub_date = _parse_review_date(item.findtext("pubDate"))
                    source_el = item.find("source")
                    publisher = (source_el.text or "").strip() if source_el is not None else ""

                    if not raw_title or not link:
                        continue

                    headline = raw_title
                    if publisher and headline.endswith(f" - {publisher}"):
                        headline = headline[: -(len(publisher) + 3)].strip()

                    references.append({
                        "claim_text": headline,
                        "claimant": "Web Intelligence (Google News RSS)",
                        "publisher_name": publisher or "Fact-Check News",
                        "publisher_url": link,
                        "rating": "Reference",
                        "review_date": pub_date,
                    })

                if references:
                    return references
            except Exception as exc:
                logger.warning("Keyless Google News RSS lookup failed (%s).", exc)
                break

        return []

    @staticmethod
    def query_fact_check_tools(query_text: str) -> dict:
        api_key = (
            getattr(settings, "FACT_CHECK_API_KEY", None)
            or os.environ.get("FACT_CHECK_API_KEY")
        )

        if not query_text or not query_text.strip():
            return {"status": "not_searched", "references": []}

        clean_query = _sanitize_search_query(query_text)
        if not api_key:
            logger.info("FACT_CHECK_API_KEY unset; running live keyless web fact-check fallback.")
            references = FactCheckService._query_google_news_rss(clean_query)
            return {"status": "found" if references else "no_results", "references": references}

        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": clean_query,
            "key": api_key,
            "languageCode": "en",
            "pageSize": 5,
        }

        retry_delays = (1, 2, 4)
        retryable_statuses = {429, 500, 502, 503, 504}

        for attempt in range(len(retry_delays) + 1):
            try:
                response = requests.get(url, params=params, timeout=10)
            except requests.RequestException as exc:
                if attempt < len(retry_delays):
                    logger.warning(
                        "Fact-check request failed; retrying (%s).",
                        type(exc).__name__,
                    )
                    time.sleep(retry_delays[attempt])
                    continue
                logger.warning(
                    "Fact-check lookup failed after retries (%s).",
                    type(exc).__name__,
                )
                references = FactCheckService._query_google_news_rss(clean_query)
                return {"status": "found" if references else "unavailable", "references": references}

            if response.status_code in retryable_statuses:
                if attempt < len(retry_delays):
                    logger.warning(
                        "Google Fact Check API returned HTTP %s; retrying.",
                        response.status_code,
                    )
                    time.sleep(retry_delays[attempt])
                    continue
                logger.warning(
                    "Google Fact Check API still returned HTTP %s after retries.",
                    response.status_code,
                )
                references = FactCheckService._query_google_news_rss(clean_query)
                return {"status": "found" if references else "unavailable", "references": references}

            if response.status_code != 200:
                # Don't log the request URL: it contains the API key.
                logger.warning(
                    "Google Fact Check API returned HTTP %s.",
                    response.status_code,
                )
                references = FactCheckService._query_google_news_rss(clean_query)
                return {"status": "found" if references else "unavailable", "references": references}

            try:
                data = response.json()
            except (ValueError, TypeError):
                logger.warning("Google Fact Check API returned invalid JSON.")
                return {"status": "unavailable", "references": []}

            references = []
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
                        "review_date": _parse_review_date(review.get("reviewDate")),
                    })

            if not references:
                references = FactCheckService._query_google_news_rss(clean_query)
            return {
                "status": "found" if references else "no_results",
                "references": references[:5],
            }

        return {"status": "unavailable", "references": []}
