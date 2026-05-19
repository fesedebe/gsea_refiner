from typing import List

import pandas as pd


def label_pathways_by_regex(
    df: pd.DataFrame,
    categories: List[str],
    cat_terms: List[str],
    col: str = "pathway",
    label_col: str = "label",
    keep_unlabeled: bool = True
) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = "Other"
    for cat, pattern in zip(categories, cat_terms):
        matched = df[col].str.contains(pattern, case=False, regex=True)
        df.loc[matched, label_col] = cat
    if not keep_unlabeled:
        df = df[df[label_col] != "Other"]
    return df
