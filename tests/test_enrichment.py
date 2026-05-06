import numpy as np
import pandas as pd

from gsea_refiner.enrichment.ks import compute_category_ks, run_category_enrichment


def _make_gsea_df(n=100, seed=42):
    rng = np.random.RandomState(seed)
    pathways = [f"PATHWAY_{i}" for i in range(n)]
    nes = rng.randn(n)
    categories = rng.choice(["immune", "cycle", "Other"], size=n, p=[0.3, 0.3, 0.4])
    return pd.DataFrame({"pathway": pathways, "NES": nes, "Category": categories})


def test_compute_category_ks_returns_expected_columns():
    df = _make_gsea_df()
    df["rank"] = range(1, len(df) + 1)
    result = compute_category_ks(df, categories=["immune", "cycle"], verbose=False)
    assert set(result.columns) == {"Category", "Freq", "pval", "ES", "signedlogp", "sign"}
    assert len(result) == 2


def test_run_category_enrichment_returns_both_tables():
    df = _make_gsea_df()
    result = run_category_enrichment(
        df, categories=["immune", "cycle"], prediction_col="Category", verbose=False
    )
    assert "pathways" in result
    assert "categories" in result
    assert "rank" in result["pathways"].columns


def test_enrichment_works_with_synthetic_predictions():
    """Confirms enrichment is classifier-agnostic — works with any prediction column."""
    df = _make_gsea_df()
    df["transformer_pred"] = df["Category"]
    result = run_category_enrichment(
        df,
        categories=["immune", "cycle"],
        prediction_col="transformer_pred",
        verbose=False,
    )
    assert len(result["categories"]) == 2
