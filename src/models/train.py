"""Phase 3 training loop for the baseline ECG classifier."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ECGDataset
from src.models.baseline_classifier import ECGClassifier1D

# Training configuration
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 10
PATIENCE = 5  # Early stopping patience
WEIGHT_DECAY = 1e-4

# Checkpoint directory (gitignored)
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"


def compute_class_weights(dataset: ECGDataset) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from the TRAIN split.

    pos_weight = (N - n_pos) / n_pos for each class.
    This up-weights the rare classes (especially HYP).

    Returns:
        Tensor of shape (5,) for NORM, MI, STTC, CD, HYP
    """
    labels = dataset.labels  # (N, 5) float32
    n_samples = labels.shape[0]
    n_pos = labels.sum(axis=0)  # (5,)
    n_neg = n_samples - n_pos

    # Avoid division by zero
    n_pos = np.maximum(n_pos, 1.0)
    pos_weight = n_neg / n_pos

    print(f"  Class counts (train): N={labels.sum(axis=0).astype(int).tolist()}")
    print(f"  pos_weights: {pos_weight.round(3).tolist()}")
    return torch.from_numpy(pos_weight.astype(np.float32))


def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders."""
    train_ds = ECGDataset("train")
    val_ds = ECGDataset("val")
    test_ds = ECGDataset("test")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
    )

    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for signals, labels, _ecg_ids in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for signals, labels, _ecg_ids in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(signals)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    # Device selection - force CPU for faster training on Apple Silicon
    device = torch.device("cpu")
    print("Using CPU (forced)")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()
    train_ds = train_loader.dataset
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Test samples:  {len(test_loader.dataset)}")

    # Class weights (computed from TRAIN only)
    pos_weight = compute_class_weights(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Model
    model = ECGClassifier1D(n_leads=12, n_classes=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'Time (s)':>10}")
    print("-" * 45)

    start_time = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Checkpoint on best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "pos_weight": pos_weight.cpu(),
                },
                BEST_MODEL_PATH,
            )
        else:
            patience_counter += 1

        print(f"{epoch:5d}  {train_loss:12.6f}  {val_loss:12.6f}  {epoch_time:10.2f}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered (no improvement for {PATIENCE} epochs)")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()