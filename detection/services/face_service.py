"""
Face detection.

Uses a pretrained YOLOv8-face model (lindevs/yolov8-face, trained on
WIDERFace) — matches the YOLOv8 requirement in your stated stack, and this
is the same detector that feeds the Tagging Layer's "objects of interest"
signal (see tagging_service.py).

Weights aren't committed to git (see .gitignore) since model files don't
belong in version control. Instead this module downloads them once on
first use and caches them under BASE_DIR/model_weights/, so a fresh clone
works with zero manual setup beyond having internet access the first time
a submission is processed.

Falls back to OpenCV's bundled Haar cascade if the download fails (offline,
GitHub unreachable, etc.) so the pipeline never hard-fails just because the
weights couldn't be fetched.
"""

import logging
from pathlib import Path

import cv2
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(settings.BASE_DIR) / "model_weights"
WEIGHTS_FILENAME = "yolov8n-face-lindevs.pt"
WEIGHTS_URL = f"https://github.com/lindevs/yolov8-face/releases/download/1.0.1/{WEIGHTS_FILENAME}"
CONFIDENCE_THRESHOLD = 0.4

_yolo_cache = {"model": None, "load_attempted": False}


def _ensure_weights_downloaded() -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = WEIGHTS_DIR / WEIGHTS_FILENAME

    if not weights_path.exists():
        logger.info("Downloading YOLOv8-face weights from %s", WEIGHTS_URL)
        response = requests.get(WEIGHTS_URL, timeout=60)
        response.raise_for_status()
        weights_path.write_bytes(response.content)

    return weights_path


def _load_yolo_face_model():
    if _yolo_cache["load_attempted"]:
        return _yolo_cache["model"]

    _yolo_cache["load_attempted"] = True
    try:
        from ultralytics import YOLO

        weights_path = _ensure_weights_downloaded()
        _yolo_cache["model"] = YOLO(str(weights_path))
        logger.info("Loaded YOLOv8-face model from %s", weights_path)
    except Exception:
        logger.warning(
            "Could not load YOLOv8-face model — falling back to OpenCV Haar cascade. "
            "Check internet access (first run needs to download weights) and that "
            "ultralytics is installed.",
            exc_info=True,
        )

    return _yolo_cache["model"]


def _detect_faces_yolo(image_path: str, model) -> list[dict]:
    results = model(image_path, verbose=False, conf=CONFIDENCE_THRESHOLD)
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.tolist()
        confs = r.boxes.conf.tolist()
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            detections.append(
                {
                    # [top, right, bottom, left] to stay consistent regardless of backend.
                    "bounding_box": [int(y1), int(x2), int(y2), int(x1)],
                    "detection_confidence": round(float(conf), 3),
                }
            )
    return detections


def _detect_faces_haar(image_path: str) -> list[dict]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces, _, weights = detector.detectMultiScale3(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), outputRejectLevels=True
    )

    results = []
    for (x, y, w, h), weight in zip(faces, weights):
        confidence = min(1.0, max(0.0, float(weight) / 10.0))
        results.append(
            {
                "bounding_box": [int(y), int(x + w), int(y + h), int(x)],
                "detection_confidence": round(confidence, 3),
            }
        )
    return results


def detect_faces(image_path: str) -> list[dict]:
    """
    Returns a list of dicts, one per detected face:
    {bounding_box: [top, right, bottom, left], detection_confidence: float}
    """
    model = _load_yolo_face_model()
    if model is not None:
        return _detect_faces_yolo(image_path, model)
    return _detect_faces_haar(image_path)