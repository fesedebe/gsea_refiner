import argparse

from gsea_refiner.classification.train import (
    DEFAULT_CLEAN_MODEL_DIR,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_DIR,
    MODELS,
    fine_tune,
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer models on labeled pathways")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS.keys()))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lr", nargs="+", type=float, default=None,
                        help="LR candidates for sweep (default: 1e-5 2e-5 5e-5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-other", action="store_true",
                        help="Include 'Other' as an 8th class (subsampled)")
    parser.add_argument("--other-sample-size", type=int, default=300,
                        help="Number of Other examples to include (default: 300)")
    parser.add_argument("--clean-model-dir", type=str, default=DEFAULT_CLEAN_MODEL_DIR,
                        help="Path to 7-class model for cleaning mislabeled Others (default: %(default)s)")
    args = parser.parse_args()

    fine_tune(
        model_names=args.models,
        model_out_dir=args.output_dir,
        lr_candidates=args.lr,
        seed=args.seed,
        include_other=args.include_other,
        other_sample_size=args.other_sample_size,
        clean_model_dir=args.clean_model_dir,
    )


if __name__ == "__main__":
    main()
