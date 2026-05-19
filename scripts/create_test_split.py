import argparse
from pathlib import Path

import pandas as pd

from gsea_refiner.evaluation.split import get_train_test_blind_split, load_blind_set


def main():
    parser = argparse.ArgumentParser(
        description="Create train/test splits and save to data/processed/."
    )
    parser.add_argument("--pool", default="data/processed/pathwaysUAP_full.csv")
    parser.add_argument("--blind", default="data/gold/others_reviewed.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-pct", type=float, default=0.9,
                        help="Training fraction (test = 1 - train_pct)")
    parser.add_argument("--output", default="data/processed/pathways_test.csv")
    parser.add_argument("--train-output", default="data/processed/pathways_train.csv")
    parser.add_argument("--full-test-output", default="data/processed/pathways_test_80_20.csv")
    args = parser.parse_args()

    train_df, matched_test_df, _ = get_train_test_blind_split(
        pool_path=args.pool, blind_path=args.blind, seed=args.seed
    )
    blind_df = load_blind_set(args.blind)

    # --- Save training set ---
    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.train_output, index=False)
    print(f"Saved training set ({len(train_df)} rows) → {args.train_output}")

    # --- Save full 80/20 test set (matched + blind) ---
    matched_copy = matched_test_df.copy()
    matched_copy["source"] = "matched"
    blind_copy = blind_df.copy()
    blind_copy["source"] = "blind"
    full_test = pd.concat([matched_copy, blind_copy], ignore_index=True)

    Path(args.full_test_output).parent.mkdir(parents=True, exist_ok=True)
    full_test.to_csv(args.full_test_output, index=False)
    print(f"Saved full test set ({len(full_test)} rows) → {args.full_test_output}")

    # --- Create 90/10 stratified subset ---
    n_train = len(train_df)
    n_total = round(n_train / args.train_pct)
    n_test = n_total - n_train
    print(f"\nTrain: {n_train} ({args.train_pct:.0%})")
    print(f"Target test: {n_test} ({1 - args.train_pct:.0%})")
    print(f"Blind: {len(blind_df)} (all included)")
    print()

    train_dist = train_df["label"].value_counts(normalize=True)
    blind_dist = blind_df["label"].value_counts()

    targets = {}
    for cat in train_dist.index:
        targets[cat] = round(n_test * train_dist[cat])

    remainder = n_test - sum(targets.values())
    if remainder != 0:
        biggest = train_dist.idxmax()
        targets[biggest] += remainder

    from_matched = {}
    for cat, target in targets.items():
        already = blind_dist.get(cat, 0)
        from_matched[cat] = max(0, target - already)

    print("Per-category breakdown:")
    print(f"  {'Category':<25s} {'Target':>6s} {'Blind':>6s} {'Matched':>7s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*7}")
    for cat in sorted(targets.keys()):
        b = blind_dist.get(cat, 0)
        print(f"  {cat:<25s} {targets[cat]:>6d} {b:>6d} {from_matched[cat]:>7d}")
    print()

    sampled_parts = []
    for cat, n in from_matched.items():
        if n == 0:
            continue
        pool = matched_test_df[matched_test_df["label"] == cat]
        if len(pool) < n:
            print(f"  WARNING: only {len(pool)} '{cat}' in matched test, need {n}")
            n = len(pool)
        sampled_parts.append(pool.sample(n=n, random_state=args.seed))

    matched_sample = pd.concat(sampled_parts, ignore_index=True)
    matched_sample["source"] = "matched"

    blind_out = blind_df.copy()
    blind_out["source"] = "blind"

    test_set = pd.concat([blind_out, matched_sample], ignore_index=True)

    print(f"Final test set: {len(test_set)} rows")
    print(f"  blind:   {len(blind_out)}")
    print(f"  matched: {len(matched_sample)}")
    print()

    print("Label distribution comparison:")
    train_pct = train_df["label"].value_counts(normalize=True).sort_index()
    test_pct = test_set["label"].value_counts(normalize=True).sort_index()
    print(f"  {'Category':<25s} {'Train%':>7s} {'Test%':>7s}")
    print(f"  {'-'*25} {'-'*7} {'-'*7}")
    for cat in sorted(set(train_pct.index) | set(test_pct.index)):
        t = train_pct.get(cat, 0)
        s = test_pct.get(cat, 0)
        print(f"  {cat:<25s} {t:>6.1%} {s:>6.1%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    test_set.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
