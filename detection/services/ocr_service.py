import logging

import pytesseract
from PIL import Image
from pytesseract import Output

logger = logging.getLogger(__name__)

MIN_WORD_CONFIDENCE = 40


def extract_text(image_path: str) -> dict:
    try:
        data = pytesseract.image_to_data(
            Image.open(image_path),
            lang="eng",
            output_type=Output.DICT,
        )
    except pytesseract.TesseractNotFoundError:
        logger.warning("OCR unavailable: Tesseract executable was not found.")
        return {"status": "unavailable", "text": "", "confidence": None}
    except (OSError, pytesseract.TesseractError):
        logger.exception("OCR failed to process the uploaded image.")
        return {"status": "unavailable", "text": "", "confidence": None}

    words = []
    confidences = []

    for word, raw_confidence in zip(data["text"], data["conf"]):
        word = word.strip()

        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            continue

        if word and confidence >= MIN_WORD_CONFIDENCE:
            words.append(word)
            confidences.append(confidence)

    text = " ".join(words).strip()
    average_confidence = (
        sum(confidences) / len(confidences) / 100
        if confidences
        else None
    )

    return {
        "status": "text_found" if text else "no_text",
        "text": text[:4000],
        "confidence": average_confidence,
    }