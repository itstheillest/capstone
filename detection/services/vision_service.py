from google.cloud import vision

def perform_web_detection(image_path: str) -> dict:
    """
    Performs web detection on a local image file using the Google Cloud Vision API.
    Extracts matching pages, domains, and full/partial image matches to feed into the database.
    """
    client = vision.ImageAnnotatorClient()

    # Read the image file from disk
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Call the web detection API
    response = client.web_detection(image=image)
    annotations = response.web_detection

    results = {
        "pages_with_matching_images": [],
        "full_matching_images": [],
        "partial_matching_images": [],
        "visually_similar_images": [],
        "best_guess_labels": []
    }

    if annotations:
        # Extract pages with matching images (crucial for URL & domain mapping)
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

    return results