"""Validation-only calibration analysis for the Phase 3 multi-label model.

The module deliberately never loads the test split.  It measures calibration
with equal-width reliability bins and fits one temperature on validation logits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import ECGDataset
from src.data.loader import SUPERCLASSES
from src.models.baseline_classifier import ECGClassifier1D

REPO = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = REPO / "checkpoints" / "best_model.pt"
QC_DIR = REPO / "notebooks" / "calibration_qc"
TEMPERATURE_PATH = REPO / "checkpoints" / "calibration_temperature.json"


def reliability_bins(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict[str, float | int]]:
    """Return bin mean confidence/empirical positive rate for sigmoid scores."""
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    y = np.asarray(labels, dtype=float).reshape(-1)
    if p.shape != y.shape:
        raise ValueError("probabilities and labels must have identical shape")
    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        lower, upper = index / n_bins, (index + 1) / n_bins
        mask = (p >= lower) & ((p < upper) if index < n_bins - 1 else (p <= upper))
        count = int(mask.sum())
        if count:
            rows.append({"bin": index, "lower": lower, "upper": upper,
                         "count": count, "mean_confidence": float(p[mask].mean()),
                         "empirical_accuracy": float(y[mask].mean())})
    return rows


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width ECE: sum_b n_b/N * |mean(p)_b - mean(y)_b|."""
    p = np.asarray(probabilities).reshape(-1)
    if not len(p):
        return 0.0
    return float(sum(row["count"] / len(p) * abs(row["mean_confidence"] - row["empirical_accuracy"])
                     for row in reliability_bins(probabilities, labels, n_bins)))


def calibration_summary(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Per-class and pooled calibration measurements."""
    return {
        "per_class": {
            name: {"ece": expected_calibration_error(probabilities[:, i], labels[:, i], n_bins),
                   "bins": reliability_bins(probabilities[:, i], labels[:, i], n_bins)}
            for i, name in enumerate(SUPERCLASSES)
        },
        "pooled": {"ece": expected_calibration_error(probabilities, labels, n_bins),
                   "bins": reliability_bins(probabilities, labels, n_bins)},
    }


def load_logits_and_labels(split: str = "val") -> tuple[np.ndarray, np.ndarray]:
    """Load the best model and infer one non-test split (VAL by default)."""
    if split == "test":
        raise ValueError("This calibration workflow is validation-only; test is evaluated by reject.py once.")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = ECGClassifier1D(n_leads=12, n_classes=len(SUPERCLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = ECGDataset(split)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    logits, labels = [], []
    with torch.no_grad():
        for signals, target, _ in loader:
            logits.append(model(signals).numpy())
            labels.append(target.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single positive T by VAL BCE, leaving model weights untouched."""
    logit_tensor = torch.as_tensor(logits, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32)
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logit_tensor / log_temperature.exp(), label_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(min=1e-3, max=1e3).item())


def save_reliability_plot(raw: dict[str, Any], calibrated: dict[str, Any]) -> Path:
    """Save five per-class curves and one pooled curve, with the ideal diagonal."""
    QC_DIR.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, name in zip(axes.flat, [*SUPERCLASSES, "pooled"]):
        for summary, label, color in ((raw, "raw", "#c44e52"), (calibrated, "temperature-scaled", "#4c72b0")):
            rows = summary[name]["bins"] if name == "pooled" else summary["per_class"][name]["bins"]
            ax.plot([r["mean_confidence"] for r in rows], [r["empirical_accuracy"] for r in rows],
                    marker="o", label=label, color=color)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        ece_raw = raw["pooled"]["ece"] if name == "pooled" else raw["per_class"][name]["ece"]
        ece_cal = calibrated["pooled"]["ece"] if name == "pooled" else calibrated["per_class"][name]["ece"]
        ax.set(title=f"{name}: ECE {ece_raw:.4f} → {ece_cal:.4f}", xlim=(0, 1), ylim=(0, 1),
               xlabel="mean predicted probability", ylabel="empirical positive rate")
        ax.legend(fontsize=8, loc="best")
    path = QC_DIR / "reliability_diagram.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    logits, labels = load_logits_and_labels("val")
    raw_probs = 1 / (1 + np.exp(-logits))
    raw = calibration_summary(raw_probs, labels)
    temperature = fit_temperature(logits, labels)
    calibrated_probs = 1 / (1 + np.exp(-(logits / temperature)))
    calibrated = calibration_summary(calibrated_probs, labels)
    path = save_reliability_plot(raw, calibrated)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    report = {"split": "val", "checkpoint": str(CHECKPOINT), "temperature": temperature,
              "raw": raw, "temperature_scaled": calibrated, "plot": str(path)}
    (QC_DIR / "calibration_metrics.json").write_text(json.dumps(report, indent=2))
    TEMPERATURE_PATH.write_text(json.dumps({"temperature": temperature, "fit_split": "val",
                                            "checkpoint_epoch": 24, "checkpoint_val_loss": 0.534156071288245}, indent=2))
    print(f"Checkpoint: {CHECKPOINT} (epoch 24, val_loss 0.534156071288245)")
    print("VAL ECE (raw):")
    for name in SUPERCLASSES:
        print(f"  {name}: {raw['per_class'][name]['ece']:.6f}")
    print(f"  pooled: {raw['pooled']['ece']:.6f}")
    print(f"Temperature: {temperature:.6f}")
    print("VAL ECE (temperature-scaled):")
    for name in SUPERCLASSES:
        print(f"  {name}: {calibrated['per_class'][name]['ece']:.6f}")
    print(f"  pooled: {calibrated['pooled']['ece']:.6f}")
    print(f"Reliability diagram: {path}")


if __name__ == "__main__":
    main()
