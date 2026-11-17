"""Evaluate bsgcasa/ai-vs-real-image-detector on a labeled folder.

    pip install torch torchvision huggingface_hub pillow
    python tools/eval_detector.py --root testset
    python tools/eval_detector.py --root testset --variant jpeg70     # robustness check

--root holds labels.csv (from build_testset.py) OR just two folders named ai/ and real/
(use that for your own images; generator/source are then blank).
"""
import argparse
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO_ID = "bsgcasa/ai-vs-real-image-detector"
NON_PHOTO_SOURCES = {"abstract", "pokemon", "wikiart"}  # art, not photographs


def load_rows(root):
    root = Path(root)
    rows = []
    if (root / "labels.csv").exists():
        with open(root / "labels.csv") as f:
            rows = [dict(r, path=str(root / r["path"])) for r in csv.DictReader(f)]
    else:
        for label in ("ai", "real"):
            for p in sorted((root / label).glob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    rows.append({"path": str(p), "label": label, "generator": "", "source": ""})
    if not rows:
        sys.exit(f"No images found under {str(root)!r}. Build the test set first "
                 "(python tools/build_testset.py --out testset) or add images to ai/ and real/ subfolders.")
    return rows


def apply_variant(img, variant):
    if variant == "jpeg70":      # re-compress: tests reliance on compression artifacts
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if variant == "downscale":   # shrink to 256px then back: tests reliance on resolution/detail
        w, h = img.size
        return img.resize((256, 256)).resize((w, h))
    return img


def load_model():
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from torchvision import models, transforms

    path = hf_hub_download(repo_id=REPO_ID, filename="checkpoint_full.pth")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # The model card loads with weights_only=False, which can run code embedded in the file.
        print("weights_only=True failed; falling back to weights_only=False. "
              "Only continue if you trust this checkpoint's source.")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    class_names = list(ckpt["class_names"])
    assert class_names == ["ai_generated", "real"], f"unexpected class order: {class_names}"
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(model.classifier[1].in_features, len(class_names)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["normalize_mean"], ckpt["normalize_std"]),
    ])
    return model, tf, float(ckpt["threshold"])


def predict_real_prob(model, tf, img):
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        return F.softmax(model(tf(img).unsqueeze(0)), dim=1)[0][1].item()  # index 1 = real


def auc(pos, neg):
    """P(a random AI image scores higher than a random real one). 0.5 = chance."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarize(rows, threshold, title):
    """rows need: label ('ai'/'real'), real_prob, generator, source. 'ai' is the positive class."""
    if not rows:
        return
    ai = [r for r in rows if r["label"] == "ai"]
    real = [r for r in rows if r["label"] == "real"]
    flagged = lambda r: r["real_prob"] < threshold  # model says AI
    tp = sum(flagged(r) for r in ai)
    tn = sum(not flagged(r) for r in real)
    print(f"\n=== {title} (threshold {threshold:.2f}) ===")
    print(f"images: {len(rows)} ({len(ai)} ai, {len(real)} real)")
    print(f"accuracy: {(tp + tn) / len(rows):.1%}")
    if ai:
        print(f"AI caught (recall):      {tp}/{len(ai)} = {tp / len(ai):.1%}")
    if real:
        print(f"real passed (specificity): {tn}/{len(real)} = {tn / len(real):.1%}")
    if ai and real:
        print(f"balanced accuracy: {(tp / len(ai) + tn / len(real)) / 2:.1%}   "
              f"AUC: {auc([1 - r['real_prob'] for r in ai], [1 - r['real_prob'] for r in real]):.3f}")
        print(f"confusion:  AI->AI {tp} | AI->real {len(ai) - tp} | real->AI {len(real) - tn} | real->real {tn}")
    for key, subset, ok in (("generator", ai, flagged), ("source", real, lambda r: not flagged(r))):
        groups = defaultdict(list)
        for r in subset:
            if r.get(key):
                groups[r[key]].append(ok(r))
        if groups:
            word = "caught" if key == "generator" else "passed"
            print(f"per {key} ({word}):")
            for k, v in sorted(groups.items()):
                print(f"  {k:14s} {sum(v)}/{len(v)} = {sum(v) / len(v):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--variant", default="none", choices=["none", "jpeg70", "downscale"])
    ap.add_argument("--threshold", type=float, default=None, help="override the checkpoint's threshold")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = load_rows(args.root)
    model, tf, ckpt_threshold = load_model()
    threshold = args.threshold if args.threshold is not None else ckpt_threshold
    
    for i, r in enumerate(rows, 1):
        img = apply_variant(Image.open(r["path"]).convert("RGB"), args.variant)
        r["real_prob"] = predict_real_prob(model, tf, img)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summarize(rows, threshold, f"ALL images [{args.variant}]")
    photos = [r for r in rows if r.get("source") not in NON_PHOTO_SOURCES]
    if len(photos) != len(rows):
        summarize(photos, threshold, f"PHOTO-ONLY, excluding {sorted(NON_PHOTO_SOURCES)} [{args.variant}]")
    print(f"\nPer-image scores saved to {args.out}")


if __name__ == "__main__":
    main()