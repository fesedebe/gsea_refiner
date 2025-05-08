import pandas as pd
import re
from typing import List

def read_file(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".txt"):
        return pd.read_table(file_path)
    else:
        raise ValueError("Unsupported file format. Only '.csv' and '.txt' are supported.")
    
def clean_gene_set_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'^[^_]*_', '', name)
    name = name.replace('_', ' ').strip()
    return name
    
def load_corpus(corpus_file: str) -> list:
    with open(corpus_file, "r") as file:
        return [line.strip() for line in file.readlines() if line.strip()]
    
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
