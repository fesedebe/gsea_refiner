from typing import Callable, Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

Predictor = Callable[[List[str]], List[str]]


def evaluate(
    predictor: Predictor,
    gold_df: pd.DataFrame,
    pathway_col: str = "pathway",
    label_col: str = "label",
) -> Dict:
    pathways = gold_df[pathway_col].tolist()
    y_true = gold_df[label_col].tolist()
    y_pred = predictor(pathways)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Predictor returned {len(y_pred)} labels for {len(y_true)} inputs"
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
