"""Sample N pathways currently labeled "Other" for hand-review.

Output is a CSV the user edits in place: fill in `reviewed_label` with one of
the existing categories or leave as "Other". Reviewed file becomes the blind
test slice for the benchmark.

Usage:
    python scripts/sample_others_for_review.py
    python scripts/sample_others_for_review.py --n 100 --seed 7
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/training/labeled_pathways_UAP.csv")
DEFAULT_OUTPUT = Path("data/gold/others_to_review.csv")
VALID_CATEGORIES = [
    "immune",
    "neuro",
    "lipid",
    "differentiation",
    "epigenetic_regulation",
    "repair",
    "cycle",
    "Other",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    others = df[df["Category"].str.strip() == "Other"].copy()
    print(f"Found {len(others)} pathways labeled 'Other' in {args.input.name}")

    if args.n > len(others):
        raise SystemExit(f"Requested {args.n} > available {len(others)}")

    sample = others.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    sample = sample.rename(columns={"Category": "original_label"})
    sample["reviewed_label"] = ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(
            f"{args.output} already exists. Move it aside before regenerating."
        )

    sample[["pathway", "original_label", "reviewed_label"]].to_csv(
        args.output, index=False
    )

    print(f"Wrote {len(sample)} rows → {args.output}")
    print("\nValid labels for the `reviewed_label` column:")
    for c in VALID_CATEGORIES:
        print(f"  - {c}")
    print("\nLeave 'Other' as-is, or replace with one of the 7 categories.")


if __name__ == "__main__":
    main()
