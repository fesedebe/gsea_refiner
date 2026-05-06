from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_DATA_PATH = Path("data/training/labeled_pathways_UAP.csv")
DEFAULT_BLIND_PATH = Path("data/gold/others_reviewed.csv")


def load_full_pool(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the labeled pool, normalizing to columns: pathway, label."""
    df = pd.read_csv(path)
    if "Category" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"Category": "label"})
    df["label"] = df["label"].astype(str).str.strip()
    return df[["pathway", "label"]].copy()


def load_blind_set(path: Path = DEFAULT_BLIND_PATH) -> pd.DataFrame:
    """Load reviewed-Others blind test slice as columns: pathway, label."""
    df = pd.read_csv(path)
    df = df.rename(columns={"reviewed_label": "label"})
    df["label"] = df["label"].astype(str).str.strip()
    return df[["pathway", "label"]].copy()


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_train_test_blind_split(
    pool_path: Path = DEFAULT_DATA_PATH,
    blind_path: Path = DEFAULT_BLIND_PATH,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (train, matched_test, blind_test).

    Removes blind pathways from the full pool before splitting, so models never
    see the stale 'Other' labels for those 200 pathways during training.
    """
    pool = load_full_pool(pool_path)
    blind = load_blind_set(blind_path)
    pool = pool[~pool["pathway"].isin(set(blind["pathway"]))].reset_index(drop=True)
    train, matched_test = stratified_split(pool, test_size=test_size, seed=seed)
    return train, matched_test, blind


def get_other_calibration_eval_split(
    pool_path: Path = DEFAULT_DATA_PATH,
    blind_path: Path = DEFAULT_BLIND_PATH,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split 'Other' examples 50/50 into calibration and evaluation sets.

    Returns (calibration_df, evaluation_df). Blind pathways are excluded
    so the same pathways aren't used for both blind-test and calibration.
    """
    pool = load_full_pool(pool_path)
    blind = load_blind_set(blind_path)
    pool = pool[~pool["pathway"].isin(set(blind["pathway"]))].reset_index(drop=True)

    others = pool[pool["label"] == "Other"].reset_index(drop=True)
    shuffled = others.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    midpoint = len(shuffled) // 2
    calibration = shuffled.iloc[:midpoint].reset_index(drop=True)
    evaluation = shuffled.iloc[midpoint:].reset_index(drop=True)
    return calibration, evaluation
