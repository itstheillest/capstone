import os
from .vision_services import perform_web_detection
from .factcheck_service import search_fact_checks

def process_image_analysis(image_path: str) -> dict:
    """
    Orchestrates Phase 2 services: runs Google Vision web detection, 
    normalizes the output, queries the Fact Check API using best guess labels,
    and includes robust error handling and fallback behaviors.
    """
    normalized_result = {
        "vision_matches": [],
        "fact_checks": [],
        "search_keywords": []
    }

    # 1. Run Vision API with error handling / quota safety
    try:
        vision_data = perform_web_detection(image_path)
    except Exception as e:
        print(f"[Warning] Google Vision API failed or quota exceeded: {e}")
        vision_data = {}

    # 2. Normalize Vision Data into DB-ready structures
    pages = vision_data.get("pages_with_matching_images", [])
    for page in pages:
        full_url = page.get("url", "")
        # Simple utility to safely extract the domain name
        domain = full_url.split("/")[2] if len(full_url.split("/")) > 2 else "unknown"
        
        normalized_result["vision_matches"].append({
            "page_url": full_url,
            "domain": domain,
            "page_title": page.get("page_title", ""),
            "similarity_score": 1.0,  # Default fallback score for web matches
            "match_type": "EXACT"
        })

    # Extract best guess keywords to feed into the Fact Check API
    keywords = vision_data.get("best_guess_labels", [])
    normalized_result["search_keywords"] = keywords

    # 3. Query Fact Check API using the top detected keywords (with fallback)
    query_string = " ".join(keywords[:3]) if keywords else "misinformation check"
    
    try:
        fact_check_data = search_fact_checks(query_string)
    except Exception as e:
        print(f"[Warning] Fact Check API failed or returned an error: {e}")
        fact_check_data = []

    # 4. Normalize Fact Check Data into DB-ready structures
    for claim in fact_check_data:
        normalized_result["fact_checks"].append({
            "claim_text": claim.get("claim_text", ""),
            "claimant": claim.get("claimant", ""),
            "publisher_name": claim.get("publisher_name", ""),
            "publisher_url": claim.get("publisher_url", ""),
            "rating": claim.get("rating", "Unrated"),
            "review_date": claim.get("review_date") or None
        })

    return normalized_result