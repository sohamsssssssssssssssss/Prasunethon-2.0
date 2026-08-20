"""Phase 6 tests for the latency benchmark harness.

Run:  .venv/bin/python -m pytest tests/test_latency.py -v -s

Tests:
  1. Smoke test: benchmark harness runs on a synthetic recording and
     returns a positive latency value under a generous ceiling.
  2. Smoke test: streaming simulation runs and returns valid stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.latency_benchmark import (  # noqa: E402
    HardwareInfo,
    InferenceResult,
    LatencyStats,
    compute_stats,
    get_hardware_info,
    load_model_and_policy,
    preprocess_raw,
    run_inference,
)
from src.models.baseline_classifier import ECGClassifier1D  # noqa: E402

FS = 500.0


def test_hardware_info_returns_valid_data() -> None:
    """get_hardware_info() should return non-empty, sensible values."""
    hw = get_hardware_info()
    print(f"\n  Hardware: {hw.cpu_model}, {hw.ram_gb}GB RAM, "
          f"{hw.n_physical_cores}P/{hw.n_logical_cores}L cores")
    print(f"  OS: {hw.os}, Python {hw.python}, PyTorch {hw.torch_version}")
    print(f"  MPS: {hw.has_mps}, CUDA: {hw.has_cuda}")
    assert hw.cpu_model != "unknown", "CPU model not detected"
    assert hw.ram_gb > 0, "RAM not detected"
    assert hw.n_logical_cores > 0, "Core count not detected"


def test_single_synthetic_inference_latency() -> None:
    """Run inference on a synthetic recording and check latency is sane.

    Sane = positive, under 5000ms (generous ceiling for a smoke test).
    """
    model, temperature, class_thresholds = load_model_and_policy()
    print(f"\n  Model loaded: {sum(p.numel() for p in model.parameters())} params")
    print(f"  Temperature: {temperature:.4f}")

    # Create a synthetic 12-lead ECG waveform (10s @ 500Hz = 5000 samples)
    rng = np.random.default_rng(42)
    raw_signal = rng.normal(0, 0.1, (12, 5000)).astype(np.float64)

    result = run_inference(model, raw_signal, temperature, class_thresholds)

    print(f"  Preprocess:  {result.preprocess_ms:.2f} ms")
    print(f"  Inference:   {result.inference_ms:.2f} ms")
    print(f"  Calibration: {result.calibration_ms:.2f} ms")
    print(f"  Reject:      {result.reject_ms:.2f} ms")
    print(f"  Total:       {result.total_ms:.2f} ms")
    print(f"  Predictions: {result.predicted}")
    print(f"  Keep:        {result.keep}")

    assert result.total_ms > 0, f"Total latency {result.total_ms}ms is not positive"
    assert result.total_ms < 5000, (
        f"Total latency {result.total_ms:.1f}ms exceeds 5000ms ceiling"
    )
    assert result.probabilities.shape == (5,), (
        f"Expected 5 probabilities, got {result.probabilities.shape}"
    )
    assert result.preprocess_ms > 0, "Preprocessing time should be positive"
    assert result.inference_ms > 0, "Inference time should be positive"


def test_compute_stats_returns_correct_values() -> None:
    """compute_stats() should return correct statistics for a known array."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    stats = compute_stats(values)
    print(f"\n  Stats for [1..10]: mean={stats.mean_ms}, median={stats.median_ms}, "
          f"p95={stats.p95_ms}, p99={stats.p99_ms}")

    assert stats.n_samples == 10
    assert abs(stats.mean_ms - 5.5) < 0.01, f"Mean should be 5.5, got {stats.mean_ms}"
    assert abs(stats.median_ms - 5.5) < 0.01, f"Median should be 5.5, got {stats.median_ms}"
    assert abs(stats.min_ms - 1.0) < 0.01
    assert abs(stats.max_ms - 10.0) < 0.01
