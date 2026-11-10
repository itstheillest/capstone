"""
Face detection service (YOLOv8-face with OpenCV Haar Cascade fallback).
"""

import logging
from pathlib import Path
import requests
from django.conf import settings

try:
    import cv2
    # Verify cv2 actually has its compiled C++ attributes and isn't an empty namespace folder
    _has_cv2 = hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")
except ImportError:
    _has_cv2 = False

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
    except ImportError:
        logger.info("ultralytics not installed — using OpenCV Haar cascade for face detection.")
    except Exception as exc:
        logger.warning("Could not load YOLOv8-face model (%s) — falling back to Haar cascade.", exc)

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

    if len(faces) == 0:
        return []

    results = []
    for (x, y, w, h), weight in zip(faces, weights):
        w_val = float(weight[0]) if hasattr(weight, "__len__") else float(weight)
        confidence = min(1.0, max(0.5, w_val / 10.0))
        results.append(
            {
                "bounding_box": [int(y), int(x + w), int(y + h), int(x)],
                "detection_confidence": round(confidence, 3),
            }
        )
    return results


def detect_faces(image_path: str) -> list[dict]:
    model = _load_yolo_face_model()
    if model is not None:
        try:
            return _detect_faces_yolo(image_path, model)
        except Exception as exc:
            logger.warning("YOLO face inference failed (%s); trying fallback.", exc)

    if _has_cv2:
        try:
            return _detect_faces_haar(image_path)
        except Exception as exc:
            logger.warning("Haar cascade inference failed (%s).", exc)

    # Return empty list rather than a hardcoded fake face when no detector is available
    return []