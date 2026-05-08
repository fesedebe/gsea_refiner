# Run confidence-threshold calibration for trained transformer models

import argparse
from pathlib import Path

from gsea_refiner.classification.calibrate import (
    calibrate,
    collect_embeddings,
    collect_scores,
    clean_calibration_others,
    train_gatekeeper,
)
from gsea_refiner.evaluation.split import (
    get_other_calibration_eval_split,
    get_train_test_blind_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--pool", default="data/training/labeled_pathways_UAP.csv")
    parser.add_argument("--blind", default="data/gold/others_reviewed.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Skip cleaning mislabeled Others from calibration set",
    )
    args = parser.parse_args()

    _, matched_test, _ = get_train_test_blind_split(
        pool_path=args.pool, blind_path=args.blind, seed=args.seed,
    )
    category_pathways = matched_test[matched_test["label"] != "Other"]["pathway"].tolist()

    cal_others, _ = get_other_calibration_eval_split(
        pool_path=args.pool, blind_path=args.blind, seed=args.seed,
    )
    other_pathways = cal_others["pathway"].tolist()

    clean_threshold = None if args.no_clean else 0.99

    print(f"Category pathways: {len(category_pathways)}")
    print(f"Other pathways:    {len(other_pathways)}")
    print(f"Cleaning:          {'off' if args.no_clean else f'remove Others with MSP > {clean_threshold}'}")
    print()

    model_root = Path(args.model_dir)
    for model_dir in sorted(model_root.iterdir()):
        final = model_dir / "final"
        if not (final.is_dir() and (final / "config.json").exists()):
            continue

        print(f"Calibrating {model_dir.name}...")
        output_path = str(final / "calibration.json")
        result = calibrate(
            model_dir=str(final),
            category_pathways=category_pathways,
            other_pathways=other_pathways,
            clean_threshold=clean_threshold,
            output_path=output_path,
        )
        cleaning = result["cleaning"]
        print(f"  scoring method: {result['scoring_method']}")
        print(f"  temperature:    {result['temperature']}")
        print(f"  threshold:      {result['threshold']:.4f}")
        print(f"  calibration F1: {result['calibration_f1']:.4f}")
        print(f"  cleaned Others: removed {cleaning['n_removed']}, "
              f"kept {cleaning['n_remaining']}")
        print(f"  category scores: mean={result['category_scores']['mean']:.4f} "
              f"std={result['category_scores']['std']:.4f}")
        print(f"  other scores:    mean={result['other_scores']['mean']:.4f} "
              f"std={result['other_scores']['std']:.4f}")
        print(f"  saved -> {output_path}")

        # Train gatekeeper on [CLS] embeddings
        print(f"  Training gatekeeper...")
        model_path = str(final)
        cat_emb = collect_embeddings(model_path, category_pathways)

        # Clean Other pathways for gatekeeper training too
        other_logits = collect_scores(model_path, other_pathways)
        if clean_threshold is not None:
            keep_mask = clean_calibration_others(other_logits, clean_threshold)
            clean_other_pathways = [
                p for p, keep in zip(other_pathways, keep_mask) if keep
            ]
        else:
            clean_other_pathways = other_pathways

        other_emb = collect_embeddings(model_path, clean_other_pathways)
        gatekeeper_path = str(final / "gatekeeper.joblib")
        gk_result = train_gatekeeper(cat_emb, other_emb, output_path=gatekeeper_path)
        print(f"  gatekeeper train F1: {gk_result['train_f1']:.4f}")
        print(f"  saved -> {gatekeeper_path}")
        print()


if __name__ == "__main__":
    main()
