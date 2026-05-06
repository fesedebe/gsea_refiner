"""GSEA-squared pipeline (Balanis et al. 2019): regex-label pathways, then run
rank-based KS enrichment per category. The enrichment step is generic and is
also reused with transformer-predicted categories (see Phase 4).
"""

import pandas as pd

from gsea_refiner.enrichment.ks import run_category_enrichment
from gsea_refiner.labeling.regex import label_pathways_by_regex


def run_gsea_sq_pipeline(input_file: str, savename: str, catmap_file: str):
    df_catmap = pd.read_csv(catmap_file)
    categories = df_catmap["Category"].tolist()
    cat_terms = df_catmap["Regex"].tolist()

    sep = "\t" if input_file.endswith(".txt") else None
    df = pd.read_csv(input_file, sep=sep, engine="python")

    df_labeled = label_pathways_by_regex(
        df, categories, cat_terms, col="pathway", label_col="Category"
    )

    return run_category_enrichment(
        df_labeled,
        categories=categories,
        prediction_col="Category",
        savename=savename,
        verbose=True,
    )


if __name__ == "__main__":
    input_file = "data/input/fGSEA_UCLAAllPatch_deseq_recur.txt"
    savename = "data/output/UCLAAllPatch_GSEAsq"
    catmap_file = "data/config/category_keywords.csv"

    run_gsea_sq_pipeline(input_file, savename, catmap_file)
