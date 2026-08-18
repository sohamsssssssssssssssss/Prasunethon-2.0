#!/usr/bin/env python
"""Train the baseline ECG classifier with per-epoch logging.

This script:
  - Loads train/val splits via ECGDataset (with waveform caching).
  - Trains with BCEWithLogitsLoss + class-weighted pos_weight.
  - Logs train_loss and val_loss per epoch.
  - Saves best checkpoint by val_loss to checkpoints/best_model.pt.
  - Writes training_history.json to checkpoints/.
  - Uses MPS (Apple Silicon) with NaN safeguards.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ECGDataset
from src.data.loader import SUPERCLASSES
from src.models.baseline_classifier import ECGClassifier1D

# Config
BATCH_SIZE = 64  # larger batch for faster epochs
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
PATIENCE = 10

REPO = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = REPO / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
HISTORY_PATH = CHECKPOINT_DIR / "training_history.json"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_pos_weight(dataset: ECGDataset) -> torch.Tensor:
    """pos_weight = n_neg / n_pos per class (from TRAIN split only)."""
    labels = dataset.labels  # (N, 5)
    n_pos = labels.sum(axis=0)
    n_neg = labels.shape[0] - n_pos
    n_pos = np.maximum(n_pos, 1.0)
    return torch.from_numpy((n_neg / n_pos).astype(np.float32))


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for signals, labels, _ in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for signals, labels, _ in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(signals)
        loss = criterion(logits, labels)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main():
    device = get_device()
    print(f"Device: {device}")

    # Datasets
    t0 = time.time()
    train_ds = ECGDataset("train")
    val_ds = ECGDataset("val")
    print(f"Datasets loaded in {time.time()-t0:.1f}s "
          f"(train={len(train_ds)}, val={len(val_ds)})")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    # Class weights
    pos_weight = compute_pos_weight(train_ds).to(device)
    print("Class pos_weight:")
    for i, c in enumerate(SUPERCLASSES):
        print(f"  {c}: {pos_weight[i]:.3f}")

    # Model
    model = ECGClassifier1D(n_leads=12, n_classes=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)

    # Training loop
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'Time':>7}")
    print("-" * 45)

    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        ep_time = time.time() - t_ep

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            marker = " *saved"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "pos_weight": pos_weight.cpu(),
            }, BEST_MODEL_PATH)
        else:
            patience_counter += 1

        entry = {"epoch": epoch, "train_loss": round(train_loss, 6),
                 "val_loss": round(val_loss, 6)}
        history.append(entry)
        print(f"{epoch:5d}  {train_loss:12.6f}  {val_loss:12.6f}  {ep_time:6.1f}s{marker}")

        # Save history after every epoch
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping (no improvement for {PATIENCE} epochs)")
            break

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")


if __name__ == "__main__":
    main()
