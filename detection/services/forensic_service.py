"""
Forensic manipulation & synthetic media analysis.

Combines:
1. Error Level Analysis (ELA) for localized splicing/tampering.
2. Spectral / High-Frequency Noise Residual & Color Correlation heuristics for AI/Diffusion detection.
3. Provenance & Container Metadata inspection (PNG chunks, C2PA/AI markers, missing camera EXIF).
4. Optional EfficientNet-B0 PyTorch classifier when trained weights are present.
"""

import io
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops
from django.conf import settings

logger = logging.getLogger(__name__)

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
CNN_WEIGHTS_PATH = Path(settings.BASE_DIR) / "model_weights" / "efficientnet_b0_forensic.pth"

ELA_QUALITY = 90
ANOMALY_BLOCK_SIZE = 32
ANOMALY_ZSCORE_THRESHOLD = 2.2

AI_SIGNATURE_KEYWORDS = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dalle",
    "comfyui",
    "automatic1111",
    "novelai",
    "firefly",
    "imagen",
    "flux",
    "bing image creator",
    "c2pa",
    "trainedalgorithmicmedia",
    "ai generated",
    "gemini",
]


def get_cnn_model():
    global _cnn_model
    if not _has_torch or not CNN_WEIGHTS_PATH.exists():
        return None
    if _cnn_model is None:
        try:
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
            state_dict = torch.load(str(CNN_WEIGHTS_PATH), map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            _cnn_model = model
        except Exception as exc:
            logger.warning("Failed to load EfficientNet forensic weights: %s", exc)
            return None
    return _cnn_model


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
                    "confidence": round(min(1.0, float(z) / 5.0), 3),
                }
            )
    return anomalies


def _analyze_synthetic_signals(image_path: str) -> tuple[float, list[dict]]:
    """
    Evaluates container provenance, missing camera sensor metadata, and
    high-frequency noise/spectral characteristics typical of AI synthesis.
    """
    score = 0.0
    synthetic_regions = []

    with Image.open(image_path) as img:
        fmt = (img.format or "").upper()
        width, height = img.size
        info_dump = " ".join(f"{k}:{v}" for k, v in (img.info or {}).items()).lower()
        raw_exif = img.getexif()

        # 1. Direct AI Generator Metadata / C2PA Signature Check
        for kw in AI_SIGNATURE_KEYWORDS:
            if kw in info_dump:
                score += 0.85
                break

        # 2. Camera Provenance Check (271=Make, 272=Model, 36867=DateTimeOriginal)
        has_camera_make = bool(raw_exif and (raw_exif.get(271) or raw_exif.get(272)))
        has_capture_date = bool(raw_exif and raw_exif.get(36867))

        if not has_camera_make and not has_capture_date:
            # Stripped or synthetic origin
            score += 0.35
            # High-res raw PNGs without camera EXIF are overwhelmingly synthetic/renders
            if fmt == "PNG" and max(width, height) >= 768:
                score += 0.20

        # 3. Common Diffusion Resolution & Aspect Ratio Check
        # Standard SDXL / Flux / Midjourney / DALL-E dimensions (multiples of 64 or 32 without camera EXIF)
        if not has_camera_make and (width % 32 == 0 or height % 32 == 0 or (width * height) >= 700000):
            score += 0.12

        # 4. High-Frequency Noise Residual & Local Variance Consistency
        rgb = np.array(img.convert("RGB"), dtype=np.float32)
        gray = rgb.mean(axis=2)

        # Compute 2D Laplacian residual to measure high-frequency sensor noise vs diffusion smoothness
        laplacian = (
            -4 * gray[1:-1, 1:-1]
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        hf_std = float(np.std(laplacian))

        # Cross-channel gradient correlation (R-G and G-B)
        rg_corr = float(np.corrcoef(rgb[:, :, 0].ravel(), rgb[:, :, 1].ravel())[0, 1])
        if np.isnan(rg_corr):
            rg_corr = 0.0

        # Synthetic renders exhibit high cross-channel structural correlation + specific HF energy bands
        if not has_camera_make:
            if rg_corr > 0.92 or (4.0 <= hf_std <= 28.0):
                score += 0.18

            # Flag high-gradient synthetic texture patches for telemetry visibility
            h, w = gray.shape
            step = ANOMALY_BLOCK_SIZE * 2
            for y in range(0, h - step, step):
                for x in range(0, w - step, step):
                    patch_std = float(np.std(laplacian[y : y + step - 2, x : x + step - 2]))
                    if patch_std > hf_std * 1.65:
                        synthetic_regions.append({
                            "box": [x, y, x + step, y + step],
                            "label": "synthetic_frequency_artifact",
                            "confidence": round(min(0.98, patch_std / (hf_std * 2.5)), 3),
                        })

    return min(0.96, score), synthetic_regions[:24]


def analyze(image_path: str) -> dict:
    """
    Runs multi-layer forensic analysis combining ELA splicing detection,
    AI/Synthetic frequency & provenance heuristics, and optional CNN weights.
    """
    diff_array = _error_level_analysis(image_path)
    ela_regions = _find_anomaly_regions(diff_array)

    total_blocks = max(
        1,
        (diff_array.shape[0] // ANOMALY_BLOCK_SIZE) * (diff_array.shape[1] // ANOMALY_BLOCK_SIZE),
    )
    ela_score = round(min(1.0, len(ela_regions) / total_blocks * 4.0), 3)

    synth_score, synth_regions = _analyze_synthetic_signals(image_path)
    combined_regions = ela_regions if len(ela_regions) >= len(synth_regions) else synth_regions

    model = get_cnn_model()
    if model is not None and transform is not None:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        ai_prob = float(probabilities[1])
        manipulation_score = round(max(ai_prob, (synth_score * 0.6) + (ela_score * 0.4)), 4)
        model_name = "EfficientNet-B0 + Spectral/ELA Hybrid"
        model_version = "1.1.0"
    else:
        manipulation_score = round(max(ela_score, synth_score), 3)
        model_name = "PixifAI Spectral + ELA Heuristic Engine"
        model_version = "v1.1"

    if model is None:
        # ELA/spectral rules are not calibrated as a classifier. They can flag
        # ordinary edits, exports, and JPEG recompression, so they must not
        # produce a definitive authentic/manipulated verdict on their own.
        verdict = "inconclusive"
    elif manipulation_score >= 0.5:
        verdict = "manipulated"
    elif manipulation_score >= 0.25:
        verdict = "inconclusive"
    else:
        verdict = "authentic"

    return {
        "model_name": model_name,
        "model_version": model_version,
        "manipulation_score": manipulation_score,
        "verdict": verdict,
        "anomaly_regions": combined_regions,
    }
