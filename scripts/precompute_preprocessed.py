"""Precompute and cache Phase 2 preprocessed signals for all splits.

This avoids re-running the expensive wavelet denoising on every epoch.
Run once before training. Outputs .npy files in data/ptbxl/preprocessed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_split
from src.data.wfdb import read_wfdb_record
from src.preprocessing.filters import preprocess

WFDB_DIR = Path(__file__).resolve().parent.parent / "data" / "ptbxl" / "wfdb"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ptbxl" / "preprocessed"
OUT_DIR.mkdir(exist_ok=True)

TARGET_LENGTH = 5000


def preprocess_and_save(split: str) -> None:
    df, _ = load_split()
    df = df[df["split"] == split].reset_index(drop=True)

    out_split_dir = OUT_DIR / split
    out_split_dir.mkdir(exist_ok=True)

    print(f"Precomputing {split} split ({len(df)} records)...")

    n_skipped = 0
    for idx, row in df.iterrows():
        ecg_id = int(row["ecg_id"])
        out_path = out_split_dir / f"{ecg_id}.npy"

        if out_path.exists():
            continue

        hea_path = WFDB_DIR / (row["filename_hr"] + ".hea")
        dat_path = WFDB_DIR / (row["filename_hr"] + ".dat")

        if not hea_path.exists() or not dat_path.exists():
            n_skipped += 1
            continue

        try:
            signal, fs, _ = read_wfdb_record(hea_path, dat_path)
            processed = preprocess(signal, fs)

            # Pad/truncate
            n_leads, n_samples = processed.shape
            if n_samples < TARGET_LENGTH:
                pad_width = TARGET_LENGTH - n_samples
                processed = np.pad(processed, ((0, 0), (0, pad_width)), mode="constant")
            elif n_samples > TARGET_LENGTH:
                processed = processed[:, :TARGET_LENGTH]

            if np.isnan(processed).any() or np.isinf(processed).any():
                n_skipped += 1
                continue

            np.save(out_path, processed.astype(np.float32))

        except Exception:
            n_skipped += 1
            continue

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")

    print(f"  Done. Skipped {n_skipped} records.")


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        preprocess_and_save(split)
    print("All splits precomputed.")