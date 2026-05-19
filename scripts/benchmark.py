import argparse
import json
from pathlib import Path

import pandas as pd

from gsea_refiner.evaluation.metrics import evaluate_from_predictions


def _flatten(method: str, slice_name: str, result: dict) -> dict:
    row = {
        "method": method,
        "slice": slice_name,
        "n": result["n"],
        "macro_f1": round(result["macro_f1"], 4),
        "accuracy": round(result["accuracy"], 4),
        "n_oov_predictions": result["n_oov_predictions"],
    }
    for label, score in result["per_class_f1"].items():
        row[f"f1_{label}"] = round(score, 4)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Compute benchmark metrics from saved predictions. No inference."
    )
    parser.add_argument("--predictions", required=True,
                        help="CSV with per-pathway predictions (from predict_test.py)")
    parser.add_argument("--test-csv", default="data/processed/pathways_test.csv",
                        help="CSV defining the test subset (from create_test_split.py)")
    parser.add_argument("--output", default="data/output/benchmark_results.csv")
    parser.add_argument("--details", default="data/output/benchmark_details.json")
    args = parser.parse_args()

    preds_df = pd.read_csv(args.predictions)
    test_df = pd.read_csv(args.test_csv)

    merged = test_df[["pathway", "label"]].merge(preds_df, on="pathway", how="inner")

    n_missing = len(test_df) - len(merged)
    if n_missing > 0:
        print(f"WARNING: {n_missing} test pathways not found in predictions file")

    print(f"Test set: {len(merged)} pathways")
    print()

    meta_cols = {"pathway", "true_label", "source", "label"}
    model_cols = [c for c in preds_df.columns if c not in meta_cols]

    y_true = merged["label"].tolist()

    rows = []
    details = {}

    for method in model_cols:
        y_pred = merged[method].tolist()
        print(f"Scoring {method} ({len(y_true)} pathways)...")
        res = evaluate_from_predictions(y_true, y_pred)
        rows.append(_flatten(method, "test", res))
        details[f"{method}__test"] = res

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    Path(args.details).parent.mkdir(parents=True, exist_ok=True)
    with open(args.details, "w") as f:
        json.dump(details, f, indent=2)

    print()
    summary_cols = ["method", "slice", "n", "macro_f1", "accuracy", "n_oov_predictions"]
    print(df[summary_cols].to_string(index=False))
    print(f"\nSaved summary → {out_path}")
    print(f"Saved details → {args.details}")


if __name__ == "__main__":
    main()
