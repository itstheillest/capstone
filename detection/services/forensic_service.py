"""
Forensic manipulation analysis.

IMPORTANT — read before treating this as your final detector: your stated
architecture calls for a trained CNN/ViT deepfake classifier plus
YOLOv8/Detectron2 region proposals. Those need labeled training data and
model weights this environment has no way to obtain or train. What's
implemented here instead is Error Level Analysis (ELA) — a genuine,
long-established forensic technique (re-save the image at a known JPEG
quality, diff against the original; authentic regions decay predictably
under re-compression, spliced/edited regions often don't, because they were
compressed at a different generation/quality than the rest of the image).

Treat this module as a real, working baseline signal — worth keeping even
after you add the trained classifier, since ELA catches a different failure
mode (localized splicing) than a whole-image CNN/ViT classifier typically
does. Swap or extend `analyze()` once your trained model is ready; the
output shape below already matches what ForensicAnalysis expects.
"""

import io
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageChops

_cnn_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        # model.load_state_dict(torch.load("detection/models/authenticity_model.pth", map_location=device))
        model.to(device)
        model.eval()
        _cnn_model = model
    return _cnn_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def analyze(image_path: str) -> dict:
    model = get_cnn_model()
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

    real_prob = float(probabilities[0])
    ai_prob = float(probabilities[1])
    is_real = real_prob > ai_prob

    return {
        "model_name": "EfficientNet-B0 Authenticity Classifier",
        "model_version": "1.0.0",
        "manipulation_score": round(ai_prob, 4),
        "verdict": "LIKELY_REAL" if is_real else "AI_GENERATED",
        "anomaly_regions": [],
    }

ELA_QUALITY = 90
# Empirically-reasonable thresholds for this heuristic, not derived from any
# labeled validation set — tune these once you have real test images with
# known ground truth, and say so explicitly if asked in your defense.
ANOMALY_BLOCK_SIZE = 32
ANOMALY_ZSCORE_THRESHOLD = 2.5


def _error_level_analysis(image_path: str) -> np.ndarray:
    original = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=ELA_QUALITY)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(original, resaved)
    return np.array(diff, dtype=np.float32)


def _find_anomaly_regions(diff_array: np.ndarray) -> list[dict]:
    """Flags image blocks whose ELA error is a statistical outlier vs the rest of the image."""
    h, w, _ = diff_array.shape
    intensity = diff_array.mean(axis=2)  # collapse RGB to a single error map

    block_scores = []
    for y in range(0, h - ANOMALY_BLOCK_SIZE, ANOMALY_BLOCK_SIZE):
        for x in range(0, w - ANOMALY_BLOCK_SIZE, ANOMALY_BLOCK_SIZE):
            block = intensity[y : y + ANOMALY_BLOCK_SIZE, x : x + ANOMALY_BLOCK_SIZE]
            block_scores.append((x, y, float(block.mean())))

    if not block_scores:
        return []

    scores = np.array([s for _, _, s in block_scores])
    mean, std = scores.mean(), scores.std()
    if std == 0:
        return []

    anomalies = []
    for x, y, score in block_scores:
        z = (score - mean) / std
        if z > ANOMALY_ZSCORE_THRESHOLD:
            anomalies.append(
                {
                    "box": [x, y, x + ANOMALY_BLOCK_SIZE, y + ANOMALY_BLOCK_SIZE],
                    "label": "ela_anomaly",
                    "confidence": round(min(1.0, z / 6.0), 3),
                }
            )
    return anomalies


def analyze(image_path: str) -> dict:
    """
    Returns a dict ready to populate ForensicAnalysis:
    model_name, model_version, manipulation_score, verdict, anomaly_regions.
    """
    diff_array = _error_level_analysis(image_path)
    anomaly_regions = _find_anomaly_regions(diff_array)

    # Manipulation score: proportion of blocks flagged as anomalous, scaled
    # into 0-1. This is a coarse heuristic, not a calibrated probability —
    # replace with your trained classifier's actual output score.
    total_blocks = max(1, (diff_array.shape[0] // ANOMALY_BLOCK_SIZE) * (diff_array.shape[1] // ANOMALY_BLOCK_SIZE))
    manipulation_score = round(min(1.0, len(anomaly_regions) / total_blocks * 3), 3)

    if manipulation_score >= 0.5:
        verdict = "manipulated"
    elif manipulation_score >= 0.2:
        verdict = "inconclusive"
    else:
        verdict = "authentic"

    return {
        "model_name": "ELA-heuristic-baseline",
        "model_version": "v0.1",
        "manipulation_score": manipulation_score,
        "verdict": verdict,
        "anomaly_regions": anomaly_regions,
    }   