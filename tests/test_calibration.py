import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gsea_refiner.classification.calibrate import (  # noqa: E402
    apply_temperature,
    max_softmax_scores,
    sweep_threshold,
)


def test_apply_temperature_sums_to_one():
    logits = np.array([[2.0, 1.0, 0.5], [0.1, 0.2, 0.3]])
    for temp in [1.0, 1.5, 2.0]:
        probs = apply_temperature(logits, temp)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_higher_temperature_flattens_distribution():
    logits = np.array([[5.0, 1.0, 0.1]])
    probs_t1 = apply_temperature(logits, 1.0)
    probs_t2 = apply_temperature(logits, 2.0)
    assert probs_t1[0].max() > probs_t2[0].max()


def test_max_softmax_scores_shape():
    logits = np.random.randn(10, 7)
    scores = max_softmax_scores(logits, temperature=1.0)
    assert scores.shape == (10,)
    assert all(0 < s <= 1 for s in scores)


def test_sweep_threshold_finds_valid_cutoff():
    cat_scores = np.array([0.9, 0.85, 0.92, 0.88, 0.95])
    other_scores = np.array([0.3, 0.4, 0.35, 0.25, 0.45])
    threshold, f1 = sweep_threshold(cat_scores, other_scores)
    assert 0.0 < threshold < 1.0
    assert f1 > 0.5


def test_sweep_threshold_perfect_separation():
    cat_scores = np.array([0.9, 0.95, 0.92])
    other_scores = np.array([0.1, 0.15, 0.12])
    threshold, f1 = sweep_threshold(cat_scores, other_scores)
    assert f1 == 1.0
