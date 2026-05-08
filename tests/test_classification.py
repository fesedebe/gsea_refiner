import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from gsea_refiner.classification.train import (  # noqa: E402
    compute_class_weights,
    prepare_data,
)


@pytest.fixture
def labeled_df():
    return pd.DataFrame({
        "pathway": [
            "HALLMARK_INFLAMMATORY_RESPONSE",
            "GOBP_CELL_CYCLE_CHECKPOINT",
            "KEGG_FATTY_ACID_METABOLISM",
            "GOBP_DNA_REPAIR",
            "REACTOME_IMMUNE_SYSTEM",
            "GOBP_LIPID_BIOSYNTHETIC_PROCESS",
            "OTHER_PATHWAY_XYZ",
        ],
        "label": [
            "immune", "cycle", "lipid", "repair",
            "immune", "lipid", "Other",
        ],
    })


def test_prepare_data_excludes_other(labeled_df):
    df, label2id, id2label = prepare_data(labeled_df)
    assert "Other" not in df["label"].values
    assert "Other" not in label2id
    assert len(df) == 6


def test_prepare_data_label_mapping(labeled_df):
    df, label2id, id2label = prepare_data(labeled_df)
    assert set(label2id.keys()) == {"immune", "cycle", "lipid", "repair"}
    assert len(id2label) == 4
    for idx, label in id2label.items():
        assert label2id[label] == idx


def test_prepare_data_adds_clean_and_id_columns(labeled_df):
    df, label2id, _ = prepare_data(labeled_df)
    assert "pathway_clean" in df.columns
    assert "label_id" in df.columns
    assert all(df["label_id"].isin(range(len(label2id))))


def test_compute_class_weights_shape():
    labels = [0, 0, 0, 1, 1, 2]
    weights = compute_class_weights(labels, num_classes=3)
    assert weights.shape == (3,)
    assert weights[2] > weights[0]


def test_compute_class_weights_balanced():
    labels = [0, 0, 1, 1, 2, 2]
    weights = compute_class_weights(labels, num_classes=3)
    torch.testing.assert_close(weights[0], weights[1])
    torch.testing.assert_close(weights[1], weights[2])
