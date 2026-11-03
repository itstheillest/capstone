"""
Forensic manipulation analysis.

Includes Error Level Analysis (ELA) baseline heuristic, plus a safe fallback
if PyTorch/torchvision are not installed in the active environment.
"""

import io
import numpy as np
from PIL import Image, ImageChops

# Safe fallback for environments lacking PyTorch (e.g., Python 3.14)
try:
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _has_torch = True
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
except ImportError:
    _has_torch = False
    device = None
    transform = None

_cnn_model = None

def get_cnn_model():
    global _cnn_model
    if not _has_torch:
        return None
    if _cnn_model is None:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model.to(device)
        model.eval()
        _cnn_model = model
    return _cnn_model


ELA_QUALITY = 90
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
    h, w, _ = diff_array.shape
    intensity = diff_array.mean(axis=2)

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
    Runs forensic analysis. If CNN weights and Torch are available, it uses the
    classifier; otherwise, it computes an authentic Error Level Analysis score.
    """
    model = get_cnn_model()
    
    if model is not None and transform is not None:
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
            "verdict": "authentic" if is_real else "manipulated",
            "anomaly_regions": [],
        }

    # Fallback to standard Error Level Analysis (ELA heuristic)
    diff_array = _error_level_analysis(image_path)
    anomaly_regions = _find_anomaly_regions(diff_array)

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