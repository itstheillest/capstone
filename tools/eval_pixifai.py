"""Score a labeled folder through PixifAI's own forensic_service.analyze().

Run from the project root (the folder with manage.py):
    python tools/eval_pixifai.py --root testset
    python tools/eval_pixifai.py --root testset --variant jpeg70
"""
import argparse
import csv
import os
import sys
import tempfile
from collections import Counter

from PIL import Image

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pixifai.settings")
import django  # noqa: E402

django.setup()

from detection.services import forensic_service as fs  # noqa: E402
from eval_detector import NON_PHOTO_SOURCES, load_rows, summarize  # noqa: E402

CNN_NAME = "EfficientNet-B0 AI-vs-Real"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--variant", default="none", choices=["none", "jpeg70"])
    ap.add_argument("--allow-heuristic", action="store_true",
                    help="keep going even if the CNN weights did not load (measures the heuristic fallback)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="only used for the heuristic fallback, which has no calibrated threshold")
    ap.add_argument("--out", default="pixifai_predictions.csv")
    args = ap.parse_args()

    rows = load_rows(args.root)
    tmp = tempfile.mkdtemp()
    for i, r in enumerate(rows, 1):
        path = r["path"]
        if args.variant == "jpeg70":  # a real re-compressed JPEG file, EXIF stripped, like a re-shared copy
            path = os.path.join(tmp, "variant.jpg")
            Image.open(r["path"]).convert("RGB").save(path, "JPEG", quality=70)
        res = fs.analyze(path)
        if i == 1 and res["model_name"] != CNN_NAME and not args.allow_heuristic:
            sys.exit(f"The CNN did not run (model_name was {res['model_name']!r}). Check that "
                     f"{fs.CNN_WEIGHTS_PATH} exists and torch imports. Re-run with --allow-heuristic "
                     "to measure the heuristic fallback instead.")
        r.update(model=res["model_name"], verdict=res["verdict"],
                 manipulation_score=res["manipulation_score"],
                 real_prob=1 - res["manipulation_score"])
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = getattr(fs, "_cnn_metadata", None)
    using_cnn = rows[0]["model"] == CNN_NAME and meta
    threshold = meta["threshold"] if using_cnn else args.threshold
    print(f"\nmodel used: {rows[0]['model']}")
    if not using_cnn:
        print("NOTE: heuristic fallback. The threshold is arbitrary here; AUC is the number that means something.")
    print("verdict strings produced:", dict(Counter(r["verdict"] for r in rows)))

    try:  # do the Tagging Layer's rules recognize these verdict strings?
        from detection.services import tagging_service
        for v in sorted({r["verdict"] for r in rows}):
            fired = [t["tag_code"] for t in (rule(1, v, 0.9) for rule in tagging_service.RULES) if t]
            print(f"  tag rules fired for verdict {v!r}: " + (", ".join(fired) if fired else "NONE   <-- no tags, so no law mapping"))
    except (ImportError, AttributeError):
        pass

    summarize(rows, threshold, f"ALL images [{args.variant}]")
    photos = [r for r in rows if r.get("source") not in NON_PHOTO_SOURCES]
    if len(photos) != len(rows):
        summarize(photos, threshold, f"PHOTO-ONLY [{args.variant}]")
    print(f"\nPer-image results saved to {args.out}")


if __name__ == "__main__":
    main()