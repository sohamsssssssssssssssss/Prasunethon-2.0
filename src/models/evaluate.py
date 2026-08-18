"""Phase 3 evaluation script for the baseline ECG classifier.

Runs the trained model ONCE on the held-out TEST split and reports:
  - Per-class precision, recall, F1 for all 5 classes (NORM, MI, STTC, CD, HYP)
  - Macro-F1 and micro-F1
  - Confusion behavior with real support (n) per class
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import ECGDataset
from src.data.loader import SUPERCLASSES
from src.models.baseline_classifier import ECGClassifier1D

# Checkpoint path
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"


def evaluate() -> None:
    # Device - force CPU for consistency
    device = torch.device("cpu")
    print("Using CPU (forced)")

    # Load test data
    test_ds = ECGDataset("test")
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False
    )
    print(f"Test samples: {len(test_ds)}")

    # Load model
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {BEST_MODEL_PATH}")

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model = ECGClassifier1D(n_leads=12, n_classes=5).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pos_weight = checkpoint.get("pos_weight")
    if pos_weight is not None:
        print(f"Loaded pos_weight from checkpoint: {pos_weight.cpu().numpy().round(3).tolist()}")

    # Run inference
    all_logits = []
    all_labels = []
    all_ecg_ids = []

    with torch.no_grad():
        for signals, labels, ecg_ids in test_loader:
            signals = signals.to(device, non_blocking=True)
            logits = model(signals)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
            all_ecg_ids.extend(ecg_ids.tolist())

    logits = torch.cat(all_logits, dim=0)  # (N, 5)
    labels = torch.cat(all_labels, dim=0)  # (N, 5)

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits).numpy()
    y_true = labels.numpy()
    y_pred = (probs >= 0.5).astype(int)

    # Per-class metrics
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION (held-out, single pass)")
    print("=" * 80)

    # Classification report with per-class metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=SUPERCLASSES,
        zero_division=0,
        digits=4,
    )
    print(report)

    # Support (true positives per class in test set)
    support = y_true.sum(axis=0).astype(int)
    print("\nSupport (n test records per class):")
    for i, cls in enumerate(SUPERCLASSES):
        print(f"  {cls}: {support[i]}")

    # Macro and micro F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    print(f"\nMacro-F1: {macro_f1:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")

    # Per-class confusion matrices (2x2 each)
    print("\nPer-class confusion matrices (TN, FP, FN, TP):")
    for i, cls in enumerate(SUPERCLASSES):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print(f"  {cls}: TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    # Multi-label stats
    n_multi_true = (y_true.sum(axis=1) > 1).sum()
    n_multi_pred = (y_pred.sum(axis=1) > 1).sum()
    print(f"\nMulti-label records: true={n_multi_true}, pred={n_multi_pred}")


if __name__ == "__main__":
    evaluate()