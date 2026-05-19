import argparse
import json
import os

import pandas as pd

DEFAULT_OUTPUT_DIR = "data/models"


def compare_models(model_dir, model_names=None):
    if model_names is None:
        candidates = sorted(
            d for d in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, d, "final", "training_metadata.json"))
        )
    else:
        candidates = model_names

    if not candidates:
        print(f"No trained models found in {model_dir}")
        return

    summary = []
    for name in candidates:
        meta_path = os.path.join(model_dir, name, "final", "training_metadata.json")
        if not os.path.isfile(meta_path):
            print(f"  Skipping {name}: no training_metadata.json found")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        best_lr = meta["best_lr"]
        lr_info = meta["lr_sweep"][str(best_lr)]
        summary.append({
            "model": name,
            "model_id": meta["model_id"],
            "best_lr": best_lr,
            "mean_f1": lr_info["mean_f1"],
            "std_f1": lr_info["std_f1"],
            **{f"fold_{i}_f1": s for i, s in enumerate(lr_info["folds"])},
        })

    if not summary:
        return

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(model_dir, "model_comparison.csv")
    os.makedirs(model_dir, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for row in summary:
        print(f"  {row['model']:12s} lr={row['best_lr']:.0e}: "
              f"{row['mean_f1']:.4f} +/- {row['std_f1']:.4f}")

    winner = max(summary, key=lambda r: r["mean_f1"])
    print(f"\nBest model: {winner['model']} ({winner['mean_f1']:.4f} +/- {winner['std_f1']:.4f})")
    print(f"Comparison saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare trained models from saved metadata")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to compare (default: auto-discover all)")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory containing trained model subdirectories")
    args = parser.parse_args()

    compare_models(args.model_dir, args.models)


if __name__ == "__main__":
    main()
