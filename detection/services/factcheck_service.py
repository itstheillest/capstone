import os
import requests
from dotenv import load_dotenv

load_dotenv()

def search_fact_checks(query_string: str, language_code: str = "en") -> list:
    """
    Queries the Google Fact Check Tools API using a search string (e.g., keywords or best guess labels).
    Returns a normalized list of claims, publisher names, URLs, and ratings.
    """
    api_key = os.getenv("FACT_CHECK_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("Missing Fact Check API key or Google API key in environment variables.")

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    params = {
        "key": api_key,
        "query": query_string,
        "languageCode": language_code
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        claims_list = []
        raw_claims = data.get("claims", [])

        for claim in raw_claims:
            claim_text = claim.get("text", "")
            claimant = claim.get("claimant", "")
            
            # Each claim can have multiple claim reviews (publisher ratings)
            for review in claim.get("claimReview", []):
                publisher = review.get("publisher", {})
                publisher_name = publisher.get("name", "")
                publisher_url = review.get("url", "")
                rating = review.get("textualRating", "")
                review_date = review.get("reviewDate", "")

                claims_list.append({
                    "claim_text": claim_text,
                    "claimant": claimant,
                    "publisher_name": publisher_name,
                    "publisher_url": publisher_url,
                    "rating": rating,
                    "review_date": review_date
                })

        return claims_list

    except requests.exceptions.RequestException as e:
        print(f"Error querying Google Fact Check API: {e}")
        return []