from functools import lru_cache
from pathlib import Path
from threading import Lock
import re

import numpy as np
from PIL import Image, ImageOps
from django.conf import settings
from ai_edge_litert.interpreter import Interpreter


MODEL_PATH = (
    Path(settings.BASE_DIR)
    / "model_weights"
    / "teachable_machine"
    / "model_unquant.tflite"
)
LABELS_PATH = MODEL_PATH.with_name("labels.txt")

# Celery will run one task at a time during this experiment.
_INFERENCE_LOCK = Lock()


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def _read_labels() -> list[str]:
    labels = []

    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        # Handles either "AI-generated" or lines such as "0 AI-generated".
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            line = parts[1]

        labels.append(line)

    normalized = [_normalize_label(label) for label in labels]
    if "ai_generated" not in normalized or "real" not in normalized:
        raise ValueError(f"Expected AI-generated and Real labels; got {labels}")

    return labels


@lru_cache(maxsize=1)
def _load_runtime():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Teachable Machine model not found: {MODEL_PATH}")
    if not LABELS_PATH.is_file():
        raise FileNotFoundError(f"Teachable Machine labels not found: {LABELS_PATH}")

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    labels = _read_labels()

    input_shape = input_detail["shape"]
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[3] != 3:
        raise ValueError(f"Unexpected model input shape: {input_shape}")

    if input_detail["dtype"] != np.float32:
        raise ValueError(
            f"Expected float32 input from floating-point export; "
            f"got {input_detail['dtype']}"
        )

    return interpreter, input_detail, output_detail, labels


def predict(image_path: str) -> dict:
    interpreter, input_detail, output_detail, labels = _load_runtime()

    _, height, width, _ = input_detail["shape"]

    with Image.open(image_path) as source:
        image = ImageOps.fit(
            source.convert("RGB"),
            (width, height),
            method=Image.Resampling.LANCZOS,
        )

    # Teachable Machine's image export uses float pixels scaled to [-1, 1].
    pixels = np.asarray(image, dtype=np.float32)
    pixels = (pixels / 127.5) - 1.0
    tensor = np.expand_dims(pixels, axis=0)

    with _INFERENCE_LOCK:
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_detail["index"])[0].astype(
            np.float32
        )

    if len(scores) != len(labels):
        raise ValueError(
            f"Model returned {len(scores)} scores for {len(labels)} labels"
        )

    # Verify this export is returning a probability vector before naming its
    # value ai_probability.
    if not np.isclose(float(scores.sum()), 1.0, atol=0.02):
        raise ValueError(f"Output does not look like probabilities: {scores}")

    normalized_labels = [_normalize_label(label) for label in labels]
    ai_index = normalized_labels.index("ai_generated")

    return {
        "model": "Teachable Machine image classifier",
        "version": "tm-export-1",
        "status": "experimental",
        "available": True,
        "predicted_class": labels[int(np.argmax(scores))],
        "ai_probability": float(scores[ai_index]),
        "class_probabilities": {
            label: float(score) for label, score in zip(labels, scores)
        },
    }