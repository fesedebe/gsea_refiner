import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gsea_refiner.evaluation.baselines import (
    make_regex_predictor,
    make_tfidf_logreg_predictor,
    make_transformer_predictor,
    make_zero_shot_predictor,
)
from gsea_refiner.evaluation.split import (
    get_other_calibration_eval_split,
    get_train_test_blind_split,
)
from gsea_refiner.preprocessing.clean import clean_gene_set_name


def _calibrate_zero_shot(classifier, candidate_labels, test_df, args):
    """Calibrate confidence threshold for zero-shot Other detection."""
    from gsea_refiner.classification.calibrate import sweep_threshold

    label_col = "true_label" if "true_label" in test_df.columns else "label"
    cat_pathways = [
        clean_gene_set_name(p)
        for p in test_df[test_df[label_col] != "Other"]["pathway"]
    ]
    cat_results = classifier(cat_pathways, candidate_labels, batch_size=16)
    if isinstance(cat_results, dict):
        cat_results = [cat_results]
    category_scores = np.array([r["scores"][0] for r in cat_results])

    cal_others, _ = get_other_calibration_eval_split(
        pool_path=args.pool, blind_path=args.blind, seed=args.seed,
    )
    other_pathways = [clean_gene_set_name(p) for p in cal_others["pathway"]]
    other_results = classifier(other_pathways, candidate_labels, batch_size=16)
    if isinstance(other_results, dict):
        other_results = [other_results]
    other_scores = np.array([r["scores"][0] for r in other_results])

    threshold, cal_f1 = sweep_threshold(category_scores, other_scores)
    print(f"  Calibrated: threshold={threshold:.4f}, cal_f1={cal_f1:.4f}")
    return threshold


def main():
    parser = argparse.ArgumentParser(
        description="Run inference for all models and save per-pathway predictions."
    )
    parser.add_argument("--pool", default="data/processed/pathwaysUAP_full.csv")
    parser.add_argument("--blind", default="data/gold/others_reviewed.csv")
    parser.add_argument("--keywords", default="data/config/category_keywords.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", default="data/models",
                        help="Parent dir containing trained model subdirectories")
    parser.add_argument("--train-csv", default="data/processed/pathways_train.csv",
                        help="Pre-saved training set (for TF-IDF). Falls back to split.")
    parser.add_argument("--test-csv", default="data/processed/pathways_test_80_20.csv",
                        help="Pre-saved full test set. Falls back to split.")
    parser.add_argument("--output", default="data/output/test_predictions.csv")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Include zero-shot NLI baseline (downloads ~1.6GB model, slow)")
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)

    if train_path.exists() and test_path.exists():
        print(f"Loading saved splits: {train_path}, {test_path}")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    else:
        print("Saved splits not found, computing from pool...")
        train_df, matched_test_df, blind_df = get_train_test_blind_split(
            pool_path=args.pool, blind_path=args.blind, seed=args.seed
        )
        matched_test_df = matched_test_df.copy()
        matched_test_df["source"] = "matched"
        blind_df = blind_df.copy()
        blind_df["source"] = "blind"
        test_df = pd.concat([matched_test_df, blind_df], ignore_index=True)

    print(f"Train:      {len(train_df):5d}")
    print(f"Test total: {len(test_df):5d}")
    print()

    pathways = test_df["pathway"].tolist()

    methods = {
        "regex_gsea_sq": make_regex_predictor(keywords_path=args.keywords),
        "tfidf_logreg": make_tfidf_logreg_predictor(train_df, seed=args.seed),
    }

    if args.zero_shot:
        from transformers import pipeline as hf_pipeline

        print("Loading zero-shot model (facebook/bart-large-mnli)...")
        zs_classifier = hf_pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
        candidate_labels = pd.read_csv(args.keywords)["Category"].tolist()

        print("Calibrating zero-shot threshold...")
        threshold = _calibrate_zero_shot(zs_classifier, candidate_labels, test_df, args)

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

    for method_name, predictor in methods.items():
        print(f"Running {method_name}...")
        preds = predictor(pathways)
        test_df[method_name] = preds

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_col = "true_label" if "true_label" in test_df.columns else "label"
    if label_col == "label":
        test_df = test_df.rename(columns={"label": "true_label"})

    cols = ["pathway", "true_label", "source"] + list(methods.keys())
    test_df[cols].to_csv(out_path, index=False)

    print(f"\nSaved {len(test_df)} predictions x {len(methods)} models -> {out_path}")


if __name__ == "__main__":
    main()
