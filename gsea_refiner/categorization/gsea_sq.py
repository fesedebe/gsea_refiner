import pandas as pd
import numpy as np
import re
import os
from typing import Union, List, Optional, Dict
from scipy.stats import ks_2samp


def load_and_rank_gsea(df_gsea: Union[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    if isinstance(df_gsea, str):
        if verbose:
            print(f"Loading GSEA results from: {df_gsea}")
        sep = '\t' if df_gsea.endswith('.txt') else None
        df_gsea = pd.read_csv(df_gsea, sep=sep, engine='python')

    df_gsea = df_gsea[df_gsea['NES'].notna()].copy()
    df_gsea.sort_values(by='NES', inplace=True)
    df_gsea['rank'] = range(1, len(df_gsea) + 1)

    if verbose:
        print(f"Loaded {len(df_gsea)} pathways with valid NES values.")

    return df_gsea


def assign_categories(df: pd.DataFrame, categories: List[str], cat_terms: List[str], verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    df['Category'] = 'Other'

    for cat, pattern in zip(categories, cat_terms):
        matched = df['pathway'].str.contains(pattern, case=False, regex=True)
        if verbose:
            print(f"{cat}: {matched.sum()} pathways matched")
        df.loc[matched, 'Category'] = cat

    df['Category'] = pd.Categorical(df['Category'], categories=categories + ['Other'])
    return df


def compute_category_ks(df: pd.DataFrame,
                         categories: List[str],
                         rep0: float = 2.2e-16,
                         signlogp_base: int = 10,
                         seed: int = 13,
                         verbose: bool = True) -> pd.DataFrame:
    results = []
    np.random.seed(seed)

    for cat in categories:
        df_cat = df[df['Category'] == cat]
        df_noncat = df[df['Category'] != cat]

        if verbose:
            print(f"** {cat} ({len(df_cat)})")

        ks_stat, pval = ks_2samp(df_cat['rank'], df_noncat['rank'])
        es = ks_stat if df_cat['rank'].mean() < df_noncat['rank'].mean() else -ks_stat
        pval = max(pval, rep0)  # avoid 0s

        signedlogp = np.sign(es) * np.abs(np.log(pval) / np.log(signlogp_base))
        sign = '+' if es < 0 else '-' if es > 0 else '0'

        results.append({
            'Category': cat,
            'Freq': len(df_cat),
            'pval': pval,
            'ES': es,
            'signedlogp': signedlogp,
            'sign': sign
        })

    return pd.DataFrame(results)


def run_gsea_squared(
    df_gsea: Union[str, pd.DataFrame],
    categories: List[str],
    cat_terms: List[str],
    rep0: float = 2.2e-16,
    signlogp_base: int = 10,
    savename: Optional[str] = None,
    seed: int = 13,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:

    df_ranked = load_and_rank_gsea(df_gsea, verbose=verbose)
    df_cat = assign_categories(df_ranked, categories, cat_terms, verbose=verbose)

    if verbose:
        print("Calculating KS statistics for categories...")
    cat_stats = compute_category_ks(
        df_cat,
        categories=categories,
        rep0=rep0,
        signlogp_base=signlogp_base,
        seed=seed,
        verbose=verbose
    )

    if savename:
        if verbose:
            print(f"Saving results to prefix: {savename}")
        df_cat.to_csv(f"{savename}_GSEAsq_pathways.csv", index=False)
        cat_stats.to_csv(f"{savename}_GSEAsq_category_kspvals.csv", index=False)

    return {
        'pathways': df_cat,
        'categories': cat_stats
    }
