import pandas as pd
import pytest

from gsea_refiner.preprocessing.filter import filter_and_weight_pathways


@pytest.fixture
def gsea_input(tmp_path):
    df = pd.DataFrame({
        "pathway": [
            "HALLMARK_INFLAMMATORY_RESPONSE",
            "GOBP_CELL_CYCLE_CHECKPOINT",
            "KEGG_FATTY_ACID_METABOLISM",
            "GOBP_DNA_REPAIR",
            "REACTOME_SIGNALING_BY_INTERLEUKINS",
        ],
        "NES": [3.1, -2.5, 1.2, 2.8, -0.5],
        "padj": [1e-8, 1e-7, 0.05, 1e-9, 0.3],
    })
    path = tmp_path / "gsea_input.csv"
    df.to_csv(path, index=False)
    return path


def test_filter_by_nes_and_pval(gsea_input, tmp_path):
    out_filtered = tmp_path / "filtered.csv"
    out_weighted = tmp_path / "weighted.csv"

    filter_and_weight_pathways(
        str(gsea_input), str(out_filtered), str(out_weighted),
        nes_threshold=2.2, pval_threshold=1e-6,
    )

    filtered = pd.read_csv(out_filtered)
    assert len(filtered) == 3
    assert set(filtered["pathway"]) == {
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "GOBP_CELL_CYCLE_CHECKPOINT",
        "GOBP_DNA_REPAIR",
    }


def test_filter_nes_only(gsea_input, tmp_path):
    out_filtered = tmp_path / "filtered.csv"
    out_weighted = tmp_path / "weighted.csv"

    filter_and_weight_pathways(
        str(gsea_input), str(out_filtered), str(out_weighted),
        nes_threshold=2.2, pval_threshold=None,
    )

    filtered = pd.read_csv(out_filtered)
    assert len(filtered) == 3
    assert "GOBP_CELL_CYCLE_CHECKPOINT" in filtered["pathway"].values


def test_no_filter(gsea_input, tmp_path):
    out_filtered = tmp_path / "filtered.csv"
    out_weighted = tmp_path / "weighted.csv"

    filter_and_weight_pathways(
        str(gsea_input), str(out_filtered), str(out_weighted),
        nes_threshold=None, pval_threshold=None,
    )

    filtered = pd.read_csv(out_filtered)
    assert len(filtered) == 5


def test_weight_scores_produced(gsea_input, tmp_path):
    out_filtered = tmp_path / "filtered.csv"
    out_weighted = tmp_path / "weighted.csv"

    filter_and_weight_pathways(
        str(gsea_input), str(out_filtered), str(out_weighted),
        nes_threshold=2.2, pval_threshold=1e-6,
    )

    weighted = pd.read_csv(out_weighted)
    assert "pathway" in weighted.columns
    assert "Weight_Score" in weighted.columns
    assert all(0 <= s <= 1 for s in weighted["Weight_Score"])


def test_missing_column_raises(tmp_path):
    df = pd.DataFrame({"gene": ["ABC"], "score": [1.0]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Column.*not found"):
        filter_and_weight_pathways(
            str(path), str(tmp_path / "f.csv"), str(tmp_path / "w.csv"),
        )
