"""Validate the reviewed Others file and report recategorization stats.

Checks:
  - all `reviewed_label` values are in the valid category set
  - no blank reviewed_label rows
  - reports how many stayed Other vs. moved to each category
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

VALID_CATEGORIES = {
    "immune",
    "neuro",
    "lipid",
    "differentiation",
    "epigenetic_regulation",
    "repair",
    "cycle",
    "Other",
}

DEFAULT_PATH = Path("data/gold/others_reviewed.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n = len(df)
    print(f"Loaded {n} rows from {args.input}")

    blanks = df["reviewed_label"].isna() | (df["reviewed_label"].astype(str).str.strip() == "")
    if blanks.any():
        raise SystemExit(f"ERROR: {blanks.sum()} rows have blank reviewed_label")

    df["reviewed_label"] = df["reviewed_label"].astype(str).str.strip()
    invalid = df[~df["reviewed_label"].isin(VALID_CATEGORIES)]
    if len(invalid):
        print("ERROR: invalid labels:")
        print(invalid[["pathway", "reviewed_label"]].to_string(index=False))
        raise SystemExit(1)

    counts = Counter(df["reviewed_label"])
    stayed_other = counts.get("Other", 0)
    moved = n - stayed_other

    print(f"\nStayed Other:   {stayed_other:4d}  ({stayed_other / n:.1%})")
    print(f"Recategorized:  {moved:4d}  ({moved / n:.1%})")

    print("\nDistribution:")
    for label in sorted(counts):
        print(f"  {label:25s} {counts[label]:4d}")


if __name__ == "__main__":
    main()
