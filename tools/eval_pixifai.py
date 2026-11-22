"""Score a labeled folder through PixifAI's full local pipeline:
EXIF extraction, face detection, forensic_service.analyze(), and tagging_service.

Run from the project root (the folder with manage.py):
    python tools/eval_pixifai.py --root ph_testset2 --out ph2_as_is_v2.csv
    python tools/eval_pixifai.py --root ph_testset2 --variant jpeg70
"""
import argparse
import csv
import os
import sys
import tempfile
from collections import Counter, defaultdict

from PIL import Image

sys.path.insert(0, os.getcwd())
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pixifai.settings")
import django  # noqa: E402

django.setup()

from detection.services import (  # noqa: E402
    exif_service,
    face_service,
    forensic_service as fs,
    tagging_service,
)
from eval_detector import NON_PHOTO_SOURCES, load_rows, summarize  # noqa: E402

CNN_NAME = "EfficientNet-B0 AI-vs-Real"


def _summarize_tags_by_group(rows: list[dict]):
    """Prints a defense-ready table showing final Tagging Layer outcomes per generator and real source."""
    ai_rows = [r for r in rows if r.get("label") == "ai"]
    real_rows = [r for r in rows if r.get("label") == "real"]

    def _print_table(title: str, subset: list[dict], group_key: str):
        if not subset:
            return
        groups = defaultdict(list)
        for r in subset:
            groups[r.get(group_key) or "unknown"].append(r)

        print(f"\n=== Tagging Layer Breakdown by {title} ===")
        header = (
            f"  {'group':<24s} {'total':>5s} | {'id_fraud':>8s} {'manip_med':>9s} | "
            f"{'caught_ai':>11s} | {'review':>6s} | {'authentic':>9s}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        tot_n = tot_id = tot_mm = tot_rev = tot_auth = 0
        for g in sorted(groups):
            items = groups[g]
            n = len(items)
            id_f = sum(1 for r in items if "identity_fraud" in r["tag_set"])
            mm = sum(1 for r in items if "manipulated_media" in r["tag_set"])
            caught = sum(
                1 for r in items if r["tag_set"] & {"identity_fraud", "manipulated_media"}
            )
            rev = sum(1 for r in items if "requires_human_review" in r["tag_set"])
            auth = sum(1 for r in items if "likely_authentic" in r["tag_set"])

            tot_n += n
            tot_id += id_f
            tot_mm += mm
            tot_rev += rev
            tot_auth += auth

            pct = (caught / n * 100) if n else 0.0
            print(
                f"  {g:<24s} {n:5d} | {id_f:8d} {mm:9d} | "
                f"{caught:4d} ({pct:4.0f}%) | {rev:6d} | {auth:9d}"
            )

        tot_caught = sum(
            1 for r in subset if r["tag_set"] & {"identity_fraud", "manipulated_media"}
        )
        tot_pct = (tot_caught / tot_n * 100) if tot_n else 0.0
        print("  " + "-" * (len(header) - 2))
        print(
            f"  {'TOTAL':<24s} {tot_n:5d} | {tot_id:8d} {tot_mm:9d} | "
            f"{tot_caught:4d} ({tot_pct:4.0f}%) | {tot_rev:6d} | {tot_auth:9d}"
        )

    _print_table("AI Generator", ai_rows, "generator")
    _print_table("Real Source", real_rows, "source")


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

        exif_data = exif_service.extract_exif(path)
        faces = face_service.detect_faces(path)
        res = fs.analyze(path)

        if i == 1 and res["model_name"] != CNN_NAME and not args.allow_heuristic:
            sys.exit(f"The CNN did not run (model_name was {res['model_name']!r}). Check that "
                     f"{fs.CNN_WEIGHTS_PATH} exists and torch imports. Re-run with --allow-heuristic "
                     "to measure the heuristic fallback instead.")

        markers = exif_data["ai_metadata_markers"]
        tags = tagging_service.generate_tags(
            image_path=path,
            face_count=len(faces),
            verdict=res["verdict"],
            manipulation_score=res["manipulation_score"],
            has_exif_discrepancy=exif_data["has_discrepancy"],
            ai_metadata_detected=exif_data["ai_metadata_detected"],
            ai_metadata_markers=markers,
        )
        tag_codes = [t["tag_code"] for t in tags]

        r.update(
            model=res["model_name"],
            verdict=res["verdict"],
            manipulation_score=res["manipulation_score"],
            real_prob=1 - res["manipulation_score"],
            face_count=len(faces),
            has_exif_discrepancy=exif_data["has_discrepancy"],
            ai_metadata_detected=exif_data["ai_metadata_detected"],
            ai_metadata_markers=";".join(markers),
            tags=";".join(tag_codes),
        )
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        r["tag_set"] = {t for t in r["tags"].split(";") if t}

    meta = getattr(fs, "_cnn_metadata", None)
    using_cnn = rows[0]["model"] == CNN_NAME and meta
    threshold = meta["threshold"] if using_cnn else args.threshold
    print(f"\nmodel used: {rows[0]['model']}")
    if not using_cnn:
        print("NOTE: heuristic fallback. The threshold is arbitrary here; AUC is the number that means something.")
    print("verdict strings produced:", dict(Counter(r["verdict"] for r in rows)))
    print("AI metadata markers detected:", sum(1 for r in rows if r["ai_metadata_detected"]), f"/ {len(rows)}")
    print("tag codes produced:", dict(Counter(t for r in rows for t in r["tag_set"])))

    _summarize_tags_by_group(rows)

    summarize(rows, threshold, f"ALL images [{args.variant}]")
    photos = [r for r in rows if r.get("source") not in NON_PHOTO_SOURCES]
    if len(photos) != len(rows):
        summarize(photos, threshold, f"PHOTO-ONLY [{args.variant}]")
    print(f"\nPer-image results saved to {args.out}")


if __name__ == "__main__":
    main()