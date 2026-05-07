import argparse

from gsea_refiner.classification.train import DEFAULT_MODELS, DEFAULT_OUTPUT_DIR, MODELS, fine_tune


def main():
    parser = argparse.ArgumentParser(description="Fine-tune transformer models on labeled pathways")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS.keys()))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lr", nargs="+", type=float, default=None,
                        help="LR candidates for sweep (default: 1e-5 2e-5 5e-5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fine_tune(
        model_names=args.models,
        model_out_dir=args.output_dir,
        lr_candidates=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
