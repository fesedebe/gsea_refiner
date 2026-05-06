import pandas as pd

from gsea_refiner.evaluation.metrics import evaluate, evaluate_other_detection
from gsea_refiner.evaluation.split import (
    get_other_calibration_eval_split,
    get_train_test_blind_split,
)


def _predict_imperfect(pathways):
    return ["immune", "cycle", "immune", "immune"]


def _predict_perfect(pathways):
    return ["immune", "cycle", "immune"]


def test_evaluate_returns_expected_keys():
    gold = pd.DataFrame({
        "pathway": ["A", "B", "C", "D"],
        "label": ["immune", "cycle", "immune", "cycle"],
    })
    result = evaluate(_predict_imperfect, gold)
    assert "macro_f1" in result
    assert "accuracy" in result
    assert "per_class_f1" in result
    assert "confusion_matrix" in result
    assert result["n"] == 4


def test_evaluate_perfect_predictor():
    gold = pd.DataFrame({
        "pathway": ["A", "B", "C"],
        "label": ["immune", "cycle", "immune"],
    })
    result = evaluate(_predict_perfect, gold)
    assert result["macro_f1"] == 1.0
    assert result["accuracy"] == 1.0


def test_evaluate_other_detection_metrics():
    y_true = ["immune", "cycle", "Other", "Other", "Other"]
    y_pred = ["immune", "Other", "Other", "Other", "immune"]

    result = evaluate_other_detection(y_true, y_pred)
    assert result["n_true_other"] == 3
    assert result["n_correct_other"] == 2
    assert result["other_detection_rate"] == 2 / 3
    assert result["n_true_category"] == 2
    assert result["n_false_other"] == 1
    assert result["false_other_rate"] == 1 / 2


def test_train_test_blind_split_no_overlap():
    train, matched, blind = get_train_test_blind_split()
    train_paths = set(train["pathway"])
    matched_paths = set(matched["pathway"])
    blind_paths = set(blind["pathway"])

    assert len(train_paths & matched_paths) == 0
    assert len(train_paths & blind_paths) == 0
    assert len(matched_paths & blind_paths) == 0


def test_other_calibration_eval_split_invariants():
    cal, evl = get_other_calibration_eval_split()

    assert len(cal) > 0
    assert len(evl) > 0
    assert set(cal["pathway"]).isdisjoint(set(evl["pathway"]))
    assert all(cal["label"] == "Other")
    assert all(evl["label"] == "Other")
