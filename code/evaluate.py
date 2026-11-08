"""Evaluate saved predictions; never train a model or generate predictions."""

import argparse

from cli_support import DEFAULT_CLASSES, classes, directory, fresh_path, run, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, help="Test ground-truth label directory")
    parser.add_argument("--pred", required=True, help="Saved Test prediction label directory")
    parser.add_argument("--classes", default=str(DEFAULT_CLASSES))
    parser.add_argument("--post", choices=("none", "confidence", "nms", "all"), default="none")
    parser.add_argument("--val-gt", help="Required for confidence/all")
    parser.add_argument("--val-pred", help="Required for confidence/all")
    parser.add_argument("--output", required=True, help="New JSON result file")
    args = parser.parse_args(argv)
    if args.post in ("confidence", "all") and (not args.val_gt or not args.val_pred):
        parser.error("--val-gt and --val-pred are required for confidence/all")
    gt = directory(args.gt)
    pred = directory(args.pred)
    protected = [gt, pred]
    val_gt = val_pred = None
    if args.post in ("confidence", "all"):
        val_gt, val_pred = directory(args.val_gt), directory(args.val_pred)
        protected.extend((val_gt, val_pred))
        if val_gt == gt or val_pred == pred:
            raise ValueError("Validation and Test directories must be distinct.")
    output = fresh_path(args.output, protected)
    names = classes(args.classes)

    from research_core import apply_g, apply_g2, ci, derive_g, score

    thresholds = derive_g(str(val_gt), str(val_pred), names) if val_gt is not None else {}
    posts = {"none": None}
    if args.post in ("confidence", "all"):
        posts["confidence"] = lambda predictions: apply_g(predictions, thresholds)
    if args.post in ("nms", "all"):
        posts["nms"] = apply_g2
    results = {}
    for label, post in posts.items():
        count, total = score(str(gt), str(pred), names, post)
        if not total:
            raise ValueError("No ground-truth TXT files found.")
        lower, upper = ci(count, total)
        results[label] = {
            "exact_images": count,
            "images": total,
            "pct": round(count / total * 100, 1),
            "clopper_pearson_95_ci": [lower, upper],
        }
    write_json(output, {
        "gt": str(gt), "pred": str(pred),
        "val_gt": str(val_gt) if val_gt else None,
        "val_pred": str(val_pred) if val_pred else None,
        "thresholds": thresholds,
        "iou_match": 0.5, "iou_nms": 0.5,
        "independent_postprocessing": True,
        "results": results,
    })
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(run(main))
