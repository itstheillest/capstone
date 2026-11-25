"""
3-tier face detection service:
  Tier 1: YOLOv8-face (model_weights/yolov8n-face-lindevs.pt)
  Tier 2: OpenCV Haar cascade (only when cv2.CascadeClassifier is available)
  Tier 3: Pure NumPy/Pillow YCbCr biometric skin/feature fallback
"""

from collections import deque
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import requests
from django.conf import settings

try:
    import cv2
    # Verify cv2 actually has its compiled C++ attributes (OpenCV 5.0 drops CascadeClassifier)
    _has_cv2 = hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")
except ImportError:
    cv2 = None
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
        logger.info("ultralytics not installed — falling back to Haar/YCbCr face detection.")
    except Exception as exc:
        logger.warning("Could not load YOLOv8-face model (%s) — falling back to Haar/YCbCr.", exc)

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


def _detect_faces_ycbcr(image_path: str) -> list[dict]:
    """
    Pure NumPy/Pillow YCbCr biometric skin-region and texture fallback.
    Rejects empty/uniform images and flat non-textured patches so no fake faces are returned.
    """
    with Image.open(image_path) as src:
        orig_w, orig_h = src.size
        if orig_w < 24 or orig_h < 24:
            return []
        rgb_img = src.convert("RGB")
        max_dim = max(orig_w, orig_h)
        if max_dim > 384:
            scale = 384.0 / max_dim
            rgb_img = rgb_img.resize(
                (max(24, int(orig_w * scale)), max(24, int(orig_h * scale))),
                Image.Resampling.BILINEAR,
            )
        ycbcr = np.array(rgb_img.convert("YCbCr"), dtype=np.float32)

    h, w, _ = ycbcr.shape
    y_chan = ycbcr[:, :, 0]
    cb_chan = ycbcr[:, :, 1]
    cr_chan = ycbcr[:, :, 2]

    # Reject solid-color or near-uniform frames immediately
    if float(np.std(y_chan)) < 10.0:
        return []

    skin_mask = (
        (y_chan > 80.0)
        & (cb_chan >= 77.0)
        & (cb_chan <= 127.0)
        & (cr_chan >= 133.0)
        & (cr_chan <= 173.0)
    )
    skin_ratio = float(np.mean(skin_mask))
    if skin_ratio < 0.015 or skin_ratio > 0.85:
        return []

    grid_rows = min(24, h)
    grid_cols = min(24, w)
    cell_h = max(1, h // grid_rows)
    cell_w = max(1, w // grid_cols)

    active = np.zeros((grid_rows, grid_cols), dtype=bool)
    for gy in range(grid_rows):
        y0 = gy * cell_h
        y1 = h if gy == grid_rows - 1 else (gy + 1) * cell_h
        for gx in range(grid_cols):
            x0 = gx * cell_w
            x1 = w if gx == grid_cols - 1 else (gx + 1) * cell_w
            if float(np.mean(skin_mask[y0:y1, x0:x1])) >= 0.35:
                active[gy, gx] = True

    visited = np.zeros_like(active)
    scale_x = orig_w / float(w)
    scale_y = orig_h / float(h)
    candidates = []

    for sy in range(grid_rows):
        for sx in range(grid_cols):
            if not active[sy, sx] or visited[sy, sx]:
                continue

            queue = deque([(sy, sx)])
            visited[sy, sx] = True
            min_gy = max_gy = sy
            min_gx = max_gx = sx

            while queue:
                cy, cx = queue.popleft()
                min_gy = min(min_gy, cy)
                max_gy = max(max_gy, cy)
                min_gx = min(min_gx, cx)
                max_gx = max(max_gx, cx)

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < grid_rows and 0 <= nx < grid_cols:
                            if active[ny, nx] and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

            y1_s = min_gy * cell_h
            y2_s = h if max_gy == grid_rows - 1 else (max_gy + 1) * cell_h
            x1_s = min_gx * cell_w
            x2_s = w if max_gx == grid_cols - 1 else (max_gx + 1) * cell_w

            top = int(round(y1_s * scale_y))
            bottom = int(round(y2_s * scale_y))
            left = int(round(x1_s * scale_x))
            right = int(round(x2_s * scale_x))

            bw = right - left
            bh = bottom - top
            if bw <= 0 or bh <= 0:
                continue

            aspect = bw / float(bh)
            area_ratio = (bw * bh) / float(orig_w * orig_h)
            if not (0.55 <= aspect <= 1.65) or not (0.015 <= area_ratio <= 0.75):
                continue

            patch_y = y_chan[y1_s:y2_s, x1_s:x2_s]
            if patch_y.size == 0:
                continue
            patch_std = float(np.std(patch_y))
            if patch_std < 12.0:
                continue

            patch_skin = float(np.mean(skin_mask[y1_s:y2_s, x1_s:x2_s]))
            confidence = min(0.75, max(0.50, 0.45 + (0.20 * patch_skin) + min(0.10, patch_std / 300.0)))
            candidates.append(
                {
                    "bounding_box": [top, right, bottom, left],
                    "detection_confidence": round(confidence, 3),
                    "_area": bw * bh,
                }
            )

    candidates.sort(key=lambda item: (item["detection_confidence"], item["_area"]), reverse=True)
    return [
        {
            "bounding_box": item["bounding_box"],
            "detection_confidence": item["detection_confidence"],
        }
        for item in candidates[:5]
    ]


def detect_faces(image_path: str) -> list[dict]:
    """Runs Tier 1 (YOLOv8-face) -> Tier 2 (OpenCV Haar) -> Tier 3 (YCbCr biometric)."""
    model = _load_yolo_face_model()
    if model is not None:
        try:
            return _detect_faces_yolo(image_path, model)
        except Exception as exc:
            logger.warning("YOLO face inference failed (%s); trying Haar fallback.", exc)

    if _has_cv2:
        try:
            return _detect_faces_haar(image_path)
        except Exception as exc:
            logger.warning("Haar cascade inference failed (%s); trying YCbCr fallback.", exc)

    try:
        return _detect_faces_ycbcr(image_path)
    except Exception as exc:
        logger.warning("YCbCr biometric face fallback failed (%s).", exc)
        return []
