"""Download a labeled AI-vs-real test set and write it to disk with a labels.csv.

    pip install datasets pillow
    python tools/build_testset.py --out testset
"""
import argparse
import csv
from pathlib import Path

DATASET = "ash12321/ai-detector-benchmark-test-data"  # 250 AI (5 generators) + 450 real (9 sources)


def export_samples(samples, out_dir):
    """samples: iterable of dicts with image (PIL), label, generator, source."""
    out = Path(out_dir)
    rows = []
    for i, s in enumerate(samples):
        img = s["image"].convert("RGB")
        label, gen, src = s["label"], s["generator"], s["source"]
        rel = f"{label}/{i:04d}_{gen}_{src}.png"
        (out / label).mkdir(parents=True, exist_ok=True)
        img.save(out / rel, format="PNG")  # lossless: pixels stay exactly as published
        rows.append({"path": rel, "label": label, "generator": gen,
                     "source": src, "width": img.width, "height": img.height})
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "generator", "source", "width", "height"])
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="testset")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(DATASET)  # splits don't matter here: we only evaluate, never train
    samples = (s for split in ds for s in ds[split])
    rows = export_samples(samples, args.out)
    n_ai = sum(r["label"] == "ai" for r in rows)
    print(f"Wrote {len(rows)} images to {args.out}/ ({n_ai} ai, {len(rows) - n_ai} real) + labels.csv")


if __name__ == "__main__":
    main()