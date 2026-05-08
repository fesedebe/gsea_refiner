from typing import Callable, Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

Predictor = Callable[[List[str]], List[str]]


def evaluate_from_predictions(
    y_true: List[str],
    y_pred: List[str],
) -> Dict:
    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Got {len(y_pred)} predictions for {len(y_true)} ground-truth labels"
        )

    true_labels = sorted(set(y_true))
    matrix_labels = sorted(set(y_true) | set(y_pred))
    n_oov = sum(1 for p in y_pred if p not in true_labels)

    per_class = f1_score(
        y_true, y_pred, average=None, labels=true_labels, zero_division=0
    )

    return {
        "n": len(y_true),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", labels=true_labels, zero_division=0
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "per_class_f1": dict(zip(true_labels, per_class.tolist())),
        "n_oov_predictions": n_oov,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=matrix_labels
        ).tolist(),
        "matrix_labels": matrix_labels,
    }


def evaluate(
    predictor: Predictor,
    gold_df: pd.DataFrame,
    pathway_col: str = "pathway",
    label_col: str = "label",
) -> Dict:
    pathways = gold_df[pathway_col].tolist()
    y_true = gold_df[label_col].tolist()
    y_pred = predictor(pathways)
    return evaluate_from_predictions(y_true, y_pred)


def evaluate_other_detection(
    y_true: List[str],
    y_pred: List[str],
    other_label: str = "Other",
) -> Dict:
    """Evaluate threshold-based 'Other' detection.

    Returns other_detection_rate (true Others correctly filtered) and
    false_other_rate (real category members incorrectly assigned Other).
    """
    true_is_other = [t == other_label for t in y_true]
    pred_is_other = [p == other_label for p in y_pred]

    n_true_other = sum(true_is_other)
    n_true_category = len(y_true) - n_true_other

    correct_other = sum(
        1 for t, p in zip(true_is_other, pred_is_other) if t and p
    )
    false_other = sum(
        1 for t, p in zip(true_is_other, pred_is_other) if not t and p
    )

    other_detection_rate = correct_other / n_true_other if n_true_other > 0 else 0.0
    false_other_rate = false_other / n_true_category if n_true_category > 0 else 0.0

    return {
        "other_detection_rate": other_detection_rate,
        "false_other_rate": false_other_rate,
        "n_true_other": n_true_other,
        "n_correct_other": correct_other,
        "n_true_category": n_true_category,
        "n_false_other": false_other,
    }
