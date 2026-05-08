# Benchmark all classifiers (regex, tfidf, transformers) on matched/blind/combined slices

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gsea_refiner.evaluation.baselines import (
    make_regex_predictor,
    make_tfidf_logreg_predictor,
    make_transformer_predictor,
    make_zero_shot_predictor,
)
from gsea_refiner.evaluation.metrics import evaluate
from gsea_refiner.evaluation.split import get_other_calibration_eval_split, get_train_test_blind_split
from gsea_refiner.preprocessing.clean import clean_gene_set_name


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="data/training/labeled_pathways_UAP.csv")
    parser.add_argument("--blind", default="data/gold/others_reviewed.csv")
    parser.add_argument("--keywords", default="data/config/category_keywords.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="data/models",
                        help="Parent dir containing trained model subdirectories")
    parser.add_argument("--output", default="data/output/benchmark_results.csv")
    parser.add_argument("--details", default="data/output/benchmark_details.json")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Include zero-shot NLI baseline (downloads ~1.6GB model, slow)")
    args = parser.parse_args()

    train_df, matched_test_df, blind_df = get_train_test_blind_split(
        pool_path=args.pool, blind_path=args.blind, seed=args.seed
    )
    print(f"Train:        {len(train_df):5d}")
    print(f"Matched test: {len(matched_test_df):5d}")
    print(f"Blind test:   {len(blind_df):5d}")
    print()

    rows = []
    details = {}

    methods = {
        "regex_gsea_sq": make_regex_predictor(keywords_path=args.keywords),
        "tfidf_logreg": make_tfidf_logreg_predictor(train_df, seed=args.seed),
    }

    if args.zero_shot:
        from gsea_refiner.classification.calibrate import sweep_threshold
        from transformers import pipeline as hf_pipeline

        print("Loading zero-shot model (facebook/bart-large-mnli)...")
        zs_classifier = hf_pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
        candidate_labels = pd.read_csv(args.keywords)["Category"].tolist()

        # Calibrate confidence threshold for Other detection
        cat_pathways = [
            clean_gene_set_name(p)
            for p in matched_test_df[matched_test_df["label"] != "Other"]["pathway"]
        ]
        cat_results = zs_classifier(cat_pathways, candidate_labels, batch_size=16)
        if isinstance(cat_results, dict):
            cat_results = [cat_results]
        category_scores = np.array([r["scores"][0] for r in cat_results])

        cal_others, _ = get_other_calibration_eval_split(
            pool_path=args.pool, blind_path=args.blind, seed=args.seed,
        )
        other_pathways = [clean_gene_set_name(p) for p in cal_others["pathway"]]
        other_results = zs_classifier(other_pathways, candidate_labels, batch_size=16)
        if isinstance(other_results, dict):
            other_results = [other_results]
        other_scores = np.array([r["scores"][0] for r in other_results])

        threshold, cal_f1 = sweep_threshold(category_scores, other_scores)
        print(f"Zero-shot calibrated: threshold={threshold:.4f}, cal_f1={cal_f1:.4f}")

        methods["zero_shot"] = make_zero_shot_predictor(
            confidence_threshold=threshold,
            keywords_path=args.keywords,
            classifier=zs_classifier,
        )

    model_dir = Path(args.model_dir)
    for name in sorted(model_dir.iterdir()) if model_dir.is_dir() else []:
        final = name / "final"
        if final.is_dir() and (final / "config.json").exists():
            methods[name.name] = make_transformer_predictor(str(final))

    combined_df = pd.concat([matched_test_df, blind_df], ignore_index=True)
    slices = {"matched": matched_test_df, "blind": blind_df, "combined": combined_df}

    for method_name, predictor in methods.items():
        for slice_name, slice_df in slices.items():
            print(f"Evaluating {method_name} on {slice_name} ({len(slice_df)})...")
            res = evaluate(predictor, slice_df)
            rows.append(_flatten(method_name, slice_name, res))
            details[f"{method_name}__{slice_name}"] = res

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
