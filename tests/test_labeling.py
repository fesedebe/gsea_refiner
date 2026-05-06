import pandas as pd

from gsea_refiner.labeling.regex import label_pathways_by_regex


def test_labels_matching_pathways():
    df = pd.DataFrame({"pathway": ["GOBP_IMMUNE_RESPONSE", "KEGG_CELL_CYCLE", "REACTOME_LIPID"]})
    result = label_pathways_by_regex(
        df,
        categories=["immune", "cycle"],
        cat_terms=["IMMUN", "CELL_CYCLE"],
        col="pathway",
    )
    assert result["label"].tolist() == ["immune", "cycle", "Other"]


def test_unlabeled_assigned_other():
    df = pd.DataFrame({"pathway": ["GOBP_UNKNOWN_THING"]})
    result = label_pathways_by_regex(df, categories=["immune"], cat_terms=["IMMUN"])
    assert result["label"].iloc[0] == "Other"


def test_keep_unlabeled_false_drops_other():
    df = pd.DataFrame({"pathway": ["GOBP_IMMUNE_RESPONSE", "GOBP_UNKNOWN"]})
    result = label_pathways_by_regex(
        df,
        categories=["immune"],
        cat_terms=["IMMUN"],
        keep_unlabeled=False,
    )
    assert len(result) == 1
    assert result["label"].iloc[0] == "immune"


def test_case_insensitive_matching():
    df = pd.DataFrame({"pathway": ["gobp_immune_response"]})
    result = label_pathways_by_regex(df, categories=["immune"], cat_terms=["IMMUN"])
    assert result["label"].iloc[0] == "immune"


def test_does_not_mutate_input():
    df = pd.DataFrame({"pathway": ["GOBP_IMMUNE_RESPONSE"]})
    original_cols = list(df.columns)
    label_pathways_by_regex(df, categories=["immune"], cat_terms=["IMMUN"])
    assert list(df.columns) == original_cols
