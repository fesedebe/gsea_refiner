import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gsea_refiner.preprocessing.clean import clean_gene_set_name

DEFAULT_TEMPERATURES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
DEFAULT_THRESHOLD_STEPS = 200
DEFAULT_SCORING_METHODS = ["msp", "energy"]
DEFAULT_CLEAN_THRESHOLD = 0.99


def collect_scores(
    model_dir: str,
    pathways: List[str],
    max_length: int = 48,
    batch_size: int = 64,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    cleaned = [clean_gene_set_name(p) for p in pathways]
    all_logits = []

    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu())

    logits = torch.cat(all_logits, dim=0)
    return logits.numpy()


def collect_embeddings(
    model_dir: str,
    pathways: List[str],
    max_length: int = 48,
    batch_size: int = 64,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, output_hidden_states=True
    )
    model.eval()

    cleaned = [clean_gene_set_name(p) for p in pathways]
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(cleaned), batch_size):
            batch = cleaned[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            outputs = model(**inputs)
            cls_embeddings = outputs.hidden_states[-1][:, 0, :]
            all_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


def train_gatekeeper(
    category_embeddings: np.ndarray,
    other_embeddings: np.ndarray,
    output_path: Optional[str] = None,
) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    X = np.concatenate([category_embeddings, other_embeddings])
    y = np.array([1] * len(category_embeddings) + [0] * len(other_embeddings))

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    train_f1 = f1_score(y, y_pred)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, path)

    return {"train_f1": train_f1, "model": clf}


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = torch.tensor(logits) / temperature
    probs = softmax(scaled, dim=-1).numpy()
    return probs


def max_softmax_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    probs = apply_temperature(logits, temperature)
    return np.max(probs, axis=1)


def energy_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = torch.tensor(logits) / temperature
    return (temperature * torch.logsumexp(scaled, dim=-1)).numpy()


def compute_scores(
    logits: np.ndarray, temperature: float, method: str = "msp"
) -> np.ndarray:
    if method == "energy":
        return energy_scores(logits, temperature)
    return max_softmax_scores(logits, temperature)


def clean_calibration_others(
    logits: np.ndarray, threshold: float = DEFAULT_CLEAN_THRESHOLD
) -> np.ndarray:
    """Return mask of likely-genuine Others (remove mislabeled real categories).

    Samples with max softmax > threshold at T=1.0 are almost certainly
    real categories mislabeled as Other.
    """
    scores = max_softmax_scores(logits, temperature=1.0)
    return scores < threshold


def sweep_threshold(
    category_scores: np.ndarray,
    other_scores: np.ndarray,
    n_steps: int = DEFAULT_THRESHOLD_STEPS,
    higher_is_category: bool = True,
) -> Tuple[float, float]:
    all_scores = np.concatenate([category_scores, other_scores])
    lo, hi = float(all_scores.min()), float(all_scores.max())
    thresholds = np.linspace(lo, hi, n_steps)

    y_true = np.array(
        [1] * len(category_scores) + [0] * len(other_scores)
    )

    best_threshold = lo
    best_f1 = 0.0

    for t in thresholds:
        if higher_is_category:
            y_pred = (all_scores >= t).astype(int)
        else:
            y_pred = (all_scores <= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    return best_threshold, best_f1


def calibrate(
    model_dir: str,
    category_pathways: List[str],
    other_pathways: List[str],
    temperatures: Optional[List[float]] = None,
    scoring_methods: Optional[List[str]] = None,
    n_threshold_steps: int = DEFAULT_THRESHOLD_STEPS,
    clean_threshold: Optional[float] = DEFAULT_CLEAN_THRESHOLD,
    output_path: Optional[str] = None,
) -> Dict:
    if temperatures is None:
        temperatures = DEFAULT_TEMPERATURES
    if scoring_methods is None:
        scoring_methods = DEFAULT_SCORING_METHODS

    cat_logits = collect_scores(model_dir, category_pathways)
    other_logits = collect_scores(model_dir, other_pathways)

    n_others_original = len(other_pathways)
    if clean_threshold is not None:
        keep_mask = clean_calibration_others(other_logits, clean_threshold)
        other_logits = other_logits[keep_mask]
        n_removed = n_others_original - len(other_logits)
    else:
        n_removed = 0

    best_result = None

    for method in scoring_methods:
        higher_is_cat = method != "entropy"
        for temp in temperatures:
            cat_scores = compute_scores(cat_logits, temp, method)
            other_scores = compute_scores(other_logits, temp, method)
            threshold, f1 = sweep_threshold(
                cat_scores, other_scores, n_threshold_steps, higher_is_cat
            )

            if best_result is None or f1 > best_result["f1"]:
                best_result = {
                    "scoring_method": method,
                    "temperature": temp,
                    "threshold": threshold,
                    "f1": f1,
                    "cat_score_mean": float(cat_scores.mean()),
                    "cat_score_std": float(cat_scores.std()),
                    "other_score_mean": float(other_scores.mean()),
                    "other_score_std": float(other_scores.std()),
                }

    result = {
        "scoring_method": best_result["scoring_method"],
        "threshold": best_result["threshold"],
        "temperature": best_result["temperature"],
        "calibration_f1": best_result["f1"],
        "category_scores": {
            "mean": best_result["cat_score_mean"],
            "std": best_result["cat_score_std"],
        },
        "other_scores": {
            "mean": best_result["other_score_mean"],
            "std": best_result["other_score_std"],
        },
        "cleaning": {
            "threshold": clean_threshold,
            "n_removed": n_removed,
            "n_remaining": len(other_logits),
        },
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    return result
