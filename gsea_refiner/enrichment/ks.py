"""Rank-based KS enrichment for predicted pathway categories.

Generic post-classification step: given a GSEA-style table with NES values and
a per-pathway category prediction (from any classifier — regex labels, BioBERT,
LLM, etc.), test whether each category's pathways are enriched at one end of
the NES-ranked list via a two-sample Kolmogorov–Smirnov test.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def load_and_rank_gsea(
    df_gsea: Union[str, pd.DataFrame], verbose: bool = True
) -> pd.DataFrame:
    if isinstance(df_gsea, str):
        if verbose:
            print(f"Loading GSEA results from: {df_gsea}")
        sep = "\t" if df_gsea.endswith(".txt") else None
        df_gsea = pd.read_csv(df_gsea, sep=sep, engine="python")

    df_gsea = df_gsea[df_gsea["NES"].notna()].copy()
    df_gsea.sort_values(by="NES", inplace=True)
    df_gsea["rank"] = range(1, len(df_gsea) + 1)

    if verbose:
        print(f"Loaded {len(df_gsea)} pathways with valid NES values.")

    return df_gsea


def compute_category_ks(
    df: pd.DataFrame,
    categories: List[str],
    category_col: str = "Category",
    rep0: float = 2.2e-16,
    signlogp_base: int = 10,
    seed: int = 13,
    verbose: bool = True,
) -> pd.DataFrame:
    results = []
    np.random.seed(seed)

    for cat in categories:
        df_cat = df[df[category_col] == cat]
        df_noncat = df[df[category_col] != cat]

        if verbose:
            print(f"** {cat} ({len(df_cat)})")

        ks_stat, pval = ks_2samp(df_cat["rank"], df_noncat["rank"])
        es = ks_stat if df_cat["rank"].mean() < df_noncat["rank"].mean() else -ks_stat
        pval = max(pval, rep0)

        signedlogp = np.sign(es) * np.abs(np.log(pval) / np.log(signlogp_base))
        sign = "+" if es < 0 else "-" if es > 0 else "0"

        results.append(
            {
                "Category": cat,
                "Freq": len(df_cat),
                "pval": pval,
                "ES": es,
                "signedlogp": signedlogp,
                "sign": sign,
            }
        )

    return pd.DataFrame(results)


def run_category_enrichment(
    df: Union[str, pd.DataFrame],
    categories: List[str],
    prediction_col: str = "Category",
    rep0: float = 2.2e-16,
    signlogp_base: int = 10,
    savename: Optional[str] = None,
    seed: int = 13,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Rank pathways by NES, then run per-category KS on `prediction_col`.

    `df` must contain `pathway`, `NES`, and `prediction_col`. The prediction
    column may come from any classifier; values outside `categories` are
    treated as `Other`.
    """
    df_ranked = load_and_rank_gsea(df, verbose=verbose)
    if prediction_col not in df_ranked.columns:
        raise ValueError(
            f"Prediction column '{prediction_col}' not found. "
            "Apply your classifier (regex, transformer, etc.) before calling this."
        )
    df_ranked[prediction_col] = pd.Categorical(
        df_ranked[prediction_col], categories=list(categories) + ["Other"]
    )

    if verbose:
        print("Calculating KS statistics for categories...")
    cat_stats = compute_category_ks(
        df_ranked,
        categories=categories,
        category_col=prediction_col,
        rep0=rep0,
        signlogp_base=signlogp_base,
        seed=seed,
        verbose=verbose,
    )

    if savename:
        if verbose:
            print(f"Saving results to prefix: {savename}")
        df_ranked.to_csv(f"{savename}_pathways.csv", index=False)
        cat_stats.to_csv(f"{savename}_category_kspvals.csv", index=False)

    return {"pathways": df_ranked, "categories": cat_stats}
