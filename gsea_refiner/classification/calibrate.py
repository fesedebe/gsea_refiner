import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from gsea_refiner.preprocessing.clean import clean_gene_set_name

DEFAULT_TEMPERATURES = [1.0, 1.5, 2.0]
DEFAULT_THRESHOLD_STEPS = 50


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


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = torch.tensor(logits) / temperature
    probs = softmax(scaled, dim=-1).numpy()
    return probs


def max_softmax_scores(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    probs = apply_temperature(logits, temperature)
    return np.max(probs, axis=1)


def sweep_threshold(
    category_scores: np.ndarray,
    other_scores: np.ndarray,
    n_steps: int = DEFAULT_THRESHOLD_STEPS,
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
        y_pred = np.array(
            [1 if s >= t else 0 for s in np.concatenate([category_scores, other_scores])]
        )
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
    n_threshold_steps: int = DEFAULT_THRESHOLD_STEPS,
    output_path: Optional[str] = None,
) -> Dict:
    if temperatures is None:
        temperatures = DEFAULT_TEMPERATURES

    cat_logits = collect_scores(model_dir, category_pathways)
    other_logits = collect_scores(model_dir, other_pathways)

    best_result = None

    for temp in temperatures:
        cat_scores = max_softmax_scores(cat_logits, temp)
        other_scores = max_softmax_scores(other_logits, temp)
        threshold, f1 = sweep_threshold(cat_scores, other_scores, n_threshold_steps)

        if best_result is None or f1 > best_result["f1"]:
            best_result = {
                "temperature": temp,
                "threshold": threshold,
                "f1": f1,
                "cat_score_mean": float(cat_scores.mean()),
                "cat_score_std": float(cat_scores.std()),
                "other_score_mean": float(other_scores.mean()),
                "other_score_std": float(other_scores.std()),
            }

    result = {
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
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2)

    return result
