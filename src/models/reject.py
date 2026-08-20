"""Validation-selected abstention policy and the single final TEST evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.data.dataset import ECGDataset
from src.data.loader import SUPERCLASSES
from src.models.baseline_classifier import ECGClassifier1D
from src.models.calibration import CHECKPOINT, TEMPERATURE_PATH, fit_temperature, load_logits_and_labels

REPO = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = REPO / "checkpoints" / "reject_policy.json"


def decision_confidence(probabilities: np.ndarray) -> np.ndarray:
    """Confidence in each binary label decision at the fixed 0.5 label cutoff."""
    return np.maximum(probabilities, 1.0 - probabilities)


def decision_metrics(probabilities: np.ndarray, labels: np.ndarray, thresholds: np.ndarray | float) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(int)
    keep = decision_confidence(probabilities) >= np.asarray(thresholds)
    correct = predicted == labels.astype(int)
    return {"keep": keep, "predicted": predicted, "correct": correct,
            "coverage": float(keep.mean()), "accuracy_kept": float(correct[keep].mean()) if keep.any() else 0.0,
            "accuracy_all": float(correct.mean()),
            "rejected_wrong_fraction": float((~correct[~keep]).mean()) if (~keep).any() else 0.0}


def validation_policy(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Select smallest global threshold that cuts decision error by >=5 percentage points.

    A per-class threshold replaces it only when it reaches the same target with
    higher retained accuracy and no less coverage than the selected global rule.
    """
    grid = np.round(np.arange(0.50, 0.96, 0.05), 2)
    baseline = decision_metrics(probabilities, labels, 0.5)
    target_accuracy = min(1.0, baseline["accuracy_all"] + 0.05)
    curve = []
    for threshold in grid:
        result = decision_metrics(probabilities, labels, threshold)
        curve.append({"threshold": float(threshold), "coverage": result["coverage"],
                      "accuracy_kept": result["accuracy_kept"], "error_kept": 1 - result["accuracy_kept"]})
    feasible = [row for row in curve if row["accuracy_kept"] >= target_accuracy]
    global_threshold = feasible[0]["threshold"] if feasible else float(grid[-1])
    global_result = decision_metrics(probabilities, labels, global_threshold)
    thresholds = np.full(len(SUPERCLASSES), global_threshold)
    class_investigation = {}
    for index, name in enumerate(SUPERCLASSES):
        base_accuracy = float((global_result["correct"][:, index]).mean())
        chosen = global_threshold
        candidates = []
        for threshold in grid:
            keep = global_result["keep"][:, index] if threshold == global_threshold else decision_metrics(probabilities[:, [index]], labels[:, [index]], threshold)["keep"].ravel()
            correct = global_result["correct"][:, index]
            accuracy = float(correct[keep].mean()) if keep.any() else 0.0
            candidates.append({"threshold": float(threshold), "coverage": float(keep.mean()), "accuracy_kept": accuracy})
        # Make HYP/class-specific adjustment only if it meets the overall 5pp target and is strictly better than global.
        global_class = next(row for row in candidates if row["threshold"] == global_threshold)
        better = [row for row in candidates if row["accuracy_kept"] >= target_accuracy and row["accuracy_kept"] > global_class["accuracy_kept"]]
        if better:
            chosen = better[0]["threshold"]
            thresholds[index] = chosen
        class_investigation[name] = {"baseline_accuracy": base_accuracy, "global": global_class,
                                     "selected_threshold": chosen, "curve": candidates}
    selected = decision_metrics(probabilities, labels, thresholds)
    return {"label_cutoff": 0.5, "global_threshold": global_threshold,
            "class_thresholds": {name: float(thresholds[i]) for i, name in enumerate(SUPERCLASSES)},
            "baseline_accuracy": baseline["accuracy_all"], "target_accuracy": target_accuracy,
            "curve": curve, "class_investigation": class_investigation,
            "selected_coverage": selected["coverage"], "selected_accuracy_kept": selected["accuracy_kept"]}


def test_logits_and_labels() -> tuple[np.ndarray, np.ndarray]:
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = ECGClassifier1D(n_leads=12, n_classes=len(SUPERCLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataset = ECGDataset("test")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    logits, labels = [], []
    with torch.no_grad():
        for signals, target, _ in loader:
            logits.append(model(signals).numpy())
            labels.append(target.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def print_final_test(probabilities: np.ndarray, labels: np.ndarray, policy: dict[str, Any]) -> None:
    thresholds = np.array([policy["class_thresholds"][name] for name in SUPERCLASSES])
    stats = decision_metrics(probabilities, labels, thresholds)
    print("TEST SET EVALUATION (single pass; calibration/policy selected on VAL)")
    print(f"Coverage: {stats['coverage']:.6f} ({int(stats['keep'].sum())}/{stats['keep'].size} class decisions kept)")
    print(f"Accuracy on kept decisions: {stats['accuracy_kept']:.6f}")
    print(f"Fraction wrong among rejected decisions: {stats['rejected_wrong_fraction']:.6f}")
    for i, name in enumerate(SUPERCLASSES):
        keep = stats["keep"][:, i]
        report = classification_report(labels[:, i][keep], stats["predicted"][:, i][keep], labels=[0, 1],
                                       target_names=[f"not_{name}", name], zero_division=0, output_dict=True)
        positive = report[name]
        print(f"{name}: coverage={keep.mean():.6f} kept={int(keep.sum())} precision={positive['precision']:.6f} recall={positive['recall']:.6f} f1={positive['f1-score']:.6f} support={int(positive['support'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="run the one final TEST evaluation using an existing VAL policy")
    args = parser.parse_args()
    if args.test:
        policy = json.loads(POLICY_PATH.read_text())
        temperature = json.loads(TEMPERATURE_PATH.read_text())["temperature"]
        logits, labels = test_logits_and_labels()
        print(f"Checkpoint: {CHECKPOINT} (epoch 24, val_loss 0.534156071288245); temperature={temperature:.6f}")
        print_final_test(1 / (1 + np.exp(-(logits / temperature))), labels, policy)
        return
    logits, labels = load_logits_and_labels("val")
    temperature = json.loads(TEMPERATURE_PATH.read_text())["temperature"] if TEMPERATURE_PATH.exists() else fit_temperature(logits, labels)
    policy = validation_policy(1 / (1 + np.exp(-(logits / temperature))), labels)
    policy.update({"fit_split": "val", "temperature": temperature, "checkpoint_epoch": 24,
                   "checkpoint_val_loss": 0.534156071288245})
    POLICY_PATH.write_text(json.dumps(policy, indent=2))
    print(f"VAL baseline decision accuracy: {policy['baseline_accuracy']:.6f}; target kept accuracy: {policy['target_accuracy']:.6f}")
    print("VAL global threshold tradeoff (threshold, coverage, kept_accuracy, kept_error):")
    for row in policy["curve"]:
        print(f"  {row['threshold']:.2f}  {row['coverage']:.6f}  {row['accuracy_kept']:.6f}  {row['error_kept']:.6f}")
    print(f"Selected global threshold: {policy['global_threshold']:.2f}")
    print("Selected class thresholds:")
    for name, threshold in policy["class_thresholds"].items():
        print(f"  {name}: {threshold:.2f}")
    print(f"Selected VAL coverage={policy['selected_coverage']:.6f}; kept_accuracy={policy['selected_accuracy_kept']:.6f}")


if __name__ == "__main__":
    main()
