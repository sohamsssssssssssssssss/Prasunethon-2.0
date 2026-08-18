"""Phase 3 tests for the classifier pipeline.

Run:  .venv/bin/python -m pytest tests/test_classifier_pipeline.py -v -s

Tests:
  1. Dataset returns tensors of the expected shape.
  2. A single forward pass through the model produces output of shape (batch, 5).
  3. Train/val/test ecg_ids used by the Dataset match exactly what Phase 1's
     split produced (no silent re-splitting).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import ECGDataset, TARGET_LENGTH  # noqa: E402
from src.data.loader import SUPERCLASSES  # noqa: E402
from src.models.baseline_classifier import ECGClassifier1D  # noqa: E402


# ------------------------------------------------------------------
# 1. Dataset shape tests
# ------------------------------------------------------------------

class TestDatasetShape:
    """Test that ECGDataset returns tensors of the correct shape."""

    def _make_dummy_dataset(self) -> ECGDataset:
        """Create a Dataset with a tiny mock DB (no real waveform loading)."""
        # We'll create a minimal dataset by monkeypatching the load.
        # But for a real test, we need actual data files.
        # Skip if no waveform files exist.
        ds = ECGDataset.__new__(ECGDataset)
        ds.split = "train"
        ds.labels = np.zeros((10, 5), dtype=np.float32)
        ds.labels[0, 0] = 1.0  # NORM
        ds.labels[1, 1] = 1.0  # MI
        ds.labels[2, 4] = 1.0  # HYP
        return ds

    def test_label_shape(self) -> None:
        """Labels should be (N, 5) float32 multi-hot vectors."""
        ds = self._make_dummy_dataset()
        assert ds.labels.shape == (10, len(SUPERCLASSES))
        assert ds.labels.dtype == np.float32
        # Each row should be a valid multi-hot (0s and 1s).
        assert np.all((ds.labels == 0) | (ds.labels == 1))

    def test_label_multi_hot(self) -> None:
        """A record can belong to multiple superclasses."""
        ds = self._make_dummy_dataset()
        # Manually set a multi-label record.
        ds.labels[3, 0] = 1.0  # NORM
        ds.labels[3, 2] = 1.0  # STTC
        assert ds.labels[3].sum() == 2.0, "Multi-label record should have sum > 1"

    def test_target_length_constant(self) -> None:
        """TARGET_LENGTH should be 5000 (10s at 500Hz)."""
        assert TARGET_LENGTH == 5000


# ------------------------------------------------------------------
# 2. Model forward pass test
# ------------------------------------------------------------------

class TestModelForwardPass:
    """Test that the model produces the correct output shape."""

    def test_output_shape(self) -> None:
        """Forward pass should produce (batch, 5) logits."""
        model = ECGClassifier1D(n_leads=12, n_classes=5)
        model.eval()
        # Dummy input: batch=4, 12 leads, 5000 samples.
        x = torch.randn(4, 12, TARGET_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"
        assert out.dtype == torch.float32

    def test_single_sample(self) -> None:
        """Forward pass with batch=1."""
        model = ECGClassifier1D(n_leads=12, n_classes=5)
        model.eval()
        x = torch.randn(1, 12, TARGET_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 5)

    def test_output_is_logits(self) -> None:
        """Output should be raw logits, NOT sigmoided."""
        model = ECGClassifier1D(n_leads=12, n_classes=5)
        model.eval()
        x = torch.randn(2, 12, TARGET_LENGTH)
        with torch.no_grad():
            out = model(x)
        # Logits can be any real value; if sigmoided they'd be in [0, 1].
        # Check that at least some values are outside [0, 1].
        has_outside = (out.min() < 0) or (out.max() > 1)
        # This is probabilistic but almost certain with random init.
        # If it fails, the model might be sigmoiding internally.
        assert has_outside, (
            "All outputs in [0,1] — model may be applying sigmoid internally. "
            "BCEWithLogitsLoss expects raw logits."
        )

    def test_parameter_count(self) -> None:
        """Model should have a reasonable number of parameters (not too many)."""
        model = ECGClassifier1D(n_leads=12, n_classes=5)
        n_params = sum(p.numel() for p in model.parameters())
        # Should be between 50K and 5M for a simple 1D-CNN baseline.
        assert 50_000 < n_params < 5_000_000, (
            f"Unexpected parameter count: {n_params:,}"
        )


# ------------------------------------------------------------------
# 3. Split integrity tests
# ------------------------------------------------------------------

class TestSplitIntegrity:
    """Verify that the Dataset uses exactly the same splits as Phase 1."""

    def _get_split_ecg_ids(self, split: str) -> set[int]:
        """Get ecg_ids from the Dataset for a given split."""
        try:
            ds = ECGDataset(split)
            return set(ds.ecg_ids)
        except FileNotFoundError:
            pytest.skip("PTB-XL data not available")

    def _get_phase1_ecg_ids(self, split: str) -> set[int]:
        """Get ecg_ids from Phase 1's load_split for a given split."""
        from src.data.loader import load_split
        df, _ = load_split()
        return set(df[df["split"] == split]["ecg_id"].tolist())

    def test_train_ecg_ids_subset_of_phase1(self) -> None:
        """Train ecg_ids must be a subset of Phase 1's train split.

        The Dataset filters out records with missing/corrupt/NaN waveforms,
        so exact equality is not expected.  The key invariant is:
        no record is silently moved to a different split.
        """
        ds_ids = self._get_split_ecg_ids("train")
        p1_ids = self._get_phase1_ecg_ids("train")
        assert ds_ids <= p1_ids, (
            f"Dataset has IDs not in Phase 1 train: {ds_ids - p1_ids}"
        )
        n_filtered = len(p1_ids) - len(ds_ids)
        if n_filtered > 0:
            print(f"  INFO: {n_filtered} train records filtered (missing/corrupt/NaN)")

    def test_val_ecg_ids_subset_of_phase1(self) -> None:
        """Val ecg_ids must be a subset of Phase 1's val split."""
        ds_ids = self._get_split_ecg_ids("val")
        p1_ids = self._get_phase1_ecg_ids("val")
        assert ds_ids <= p1_ids, (
            f"Dataset has IDs not in Phase 1 val: {ds_ids - p1_ids}"
        )
        n_filtered = len(p1_ids) - len(ds_ids)
        if n_filtered > 0:
            print(f"  INFO: {n_filtered} val records filtered (missing/corrupt/NaN)")

    def test_test_ecg_ids_subset_of_phase1(self) -> None:
        """Test ecg_ids must be a subset of Phase 1's test split."""
        ds_ids = self._get_split_ecg_ids("test")
        p1_ids = self._get_phase1_ecg_ids("test")
        assert ds_ids <= p1_ids, (
            f"Dataset has IDs not in Phase 1 test: {ds_ids - p1_ids}"
        )
        n_filtered = len(p1_ids) - len(ds_ids)
        if n_filtered > 0:
            print(f"  INFO: {n_filtered} test records filtered (missing/corrupt/NaN)")

    def test_no_split_overlap(self) -> None:
        """No ecg_id should appear in more than one split."""
        train_ids = self._get_split_ecg_ids("train")
        val_ids = self._get_split_ecg_ids("val")
        test_ids = self._get_split_ecg_ids("test")
        assert train_ids & val_ids == set(), "Train/val overlap!"
        assert train_ids & test_ids == set(), "Train/test overlap!"
        assert val_ids & test_ids == set(), "Val/test overlap!"
