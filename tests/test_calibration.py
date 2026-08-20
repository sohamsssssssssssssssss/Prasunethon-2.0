"""Focused unit tests for calibration and abstention helpers."""

import json
from pathlib import Path

import numpy as np

from src.models.calibration import expected_calibration_error
from src.models.reject import POLICY_PATH, decision_metrics


def test_ece_is_zero_for_perfectly_calibrated_synthetic_predictor() -> None:
    # Each confidence value has exactly its stated positive frequency.
    probabilities = np.array([0.25] * 4 + [0.75] * 4)
    labels = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    ece = expected_calibration_error(probabilities, labels, n_bins=10)
    print(f"perfectly calibrated synthetic ECE={ece:.6f}")
    assert ece == 0.0


def test_reject_reduces_kept_error_rate() -> None:
    probabilities = np.array([[0.99], [0.90], [0.60], [0.51], [0.49], [0.40], [0.10], [0.01]])
    labels = np.array([[1], [1], [0], [0], [1], [1], [0], [0]])
    baseline = decision_metrics(probabilities, labels, 0.50)
    rejected = decision_metrics(probabilities, labels, 0.80)
    baseline_error = 1 - baseline["accuracy_all"]
    kept_error = 1 - rejected["accuracy_kept"]
    print(f"baseline_error={baseline_error:.6f}; kept_error={kept_error:.6f}; coverage={rejected['coverage']:.6f}")
    assert kept_error < baseline_error


def test_validation_policy_reduces_measured_kept_error() -> None:
    """Regression check on the real VAL split; TEST is never loaded here."""
    from src.models.calibration import load_logits_and_labels

    policy = json.loads(Path(POLICY_PATH).read_text())
    logits, labels = load_logits_and_labels("val")
    probabilities = 1 / (1 + np.exp(-(logits / policy["temperature"])))
    thresholds = np.array([policy["class_thresholds"][name] for name in ["NORM", "MI", "STTC", "CD", "HYP"]])
    baseline = decision_metrics(probabilities, labels, 0.5)
    selected = decision_metrics(probabilities, labels, thresholds)
    baseline_error = 1 - baseline["accuracy_all"]
    kept_error = 1 - selected["accuracy_kept"]
    print(f"VAL measured baseline_error={baseline_error:.6f}; kept_error={kept_error:.6f}; coverage={selected['coverage']:.6f}")
    assert kept_error < baseline_error
