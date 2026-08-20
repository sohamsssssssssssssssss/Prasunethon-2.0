"""Phase 5 tests for the drift detection PoC.

Run:  .venv/bin/python -m pytest tests/test_drift.py -v -s

Tests:
  1. Feature extraction on a synthetic clean ECG-like signal with known
     RR interval returns the expected value within tolerance.
  2. Feature extraction on a synthetic ECG with a wider QRS complex
     returns a QRS width in a physiologically plausible range (single-lead
     preprocessed: 25-80 ms).
  3. Drift flag triggers on a synthetic case with an obvious injected
     feature shift.
  4. Drift flag does NOT trigger on a synthetic case with no shift.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.drift import DriftConfig, compute_z_scores  # noqa: E402
from src.preprocessing.beat_features import (  # noqa: E402
    BeatFeatures,
    detect_rpeaks,
    extract_beat_features,
)

FS = 500.0


# ------------------------------------------------------------------
# Helper: generate a synthetic clean ECG-like signal
# ------------------------------------------------------------------

def _make_synthetic_ecg(
    heart_rate_bpm: float = 72.0,
    duration_s: float = 10.0,
    fs: float = FS,
    r_peak_height_mv: float = 1.5,
    qrs_width_samples: int = 8,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Generate a synthetic clean ECG-like signal with known parameters.

    Creates a simple QRS-complex-like signal with:
    - Regular R-peaks at the specified heart rate.
    - Gaussian-shaped QRS complexes.
    - Tiny noise floor.

    Returns 1-D array in mV.
    """
    rng = np.random.default_rng(42)
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs
    signal = np.full(n_samples, 0.05, dtype=np.float64)  # baseline

    # R-peak timing
    rr_s = 60.0 / heart_rate_bpm
    r_peak_times = np.arange(0, duration_s, rr_s)

    for t_r in r_peak_times:
        idx_r = int(t_r * fs)
        if idx_r >= n_samples:
            break
        # Gaussian QRS complex
        half_w = qrs_width_samples
        lo = max(0, idx_r - half_w)
        hi = min(n_samples, idx_r + half_w + 1)
        x_local = np.arange(lo, hi) - idx_r
        qrs = r_peak_height_mv * np.exp(-0.5 * (x_local / (qrs_width_samples / 3)) ** 2)
        signal[lo:hi] += qrs

    # Add small noise
    signal += rng.normal(0, noise_std, n_samples)

    return signal


def _make_synthetic_ecg_wide_qrs(
    heart_rate_bpm: float = 72.0,
    duration_s: float = 10.0,
    fs: float = FS,
    r_peak_height_mv: float = 1.5,
    qrs_width_ms: float = 50.0,  # Target QRS width in ms
    noise_std: float = 0.01,
) -> np.ndarray:
    """Generate a synthetic ECG with a wider, more realistic QRS complex.

    Uses a wider Gaussian (sigma ~ qrs_width_ms/4) to produce a QRS
    complex that yields a tangent-method width in the ~25-80 ms range
    for single-lead preprocessed signals.
    """
    rng = np.random.default_rng(123)
    n_samples = int(duration_s * fs)
    signal = np.full(n_samples, 0.05, dtype=np.float64)

    rr_s = 60.0 / heart_rate_bpm
    r_peak_times = np.arange(0, duration_s, rr_s)

    # Convert target QRS width to Gaussian sigma
    # Tangent method on Gaussian gives width ~ 2.3 * sigma (empirical)
    sigma_samples = (qrs_width_ms / 1000.0 * fs) / 2.3
    half_w = int(4 * sigma_samples)  # 4 sigma coverage

    for t_r in r_peak_times:
        idx_r = int(t_r * fs)
        if idx_r >= n_samples:
            break
        lo = max(0, idx_r - half_w)
        hi = min(n_samples, idx_r + half_w + 1)
        x_local = np.arange(lo, hi) - idx_r
        qrs = r_peak_height_mv * np.exp(-0.5 * (x_local / sigma_samples) ** 2)
        signal[lo:hi] += qrs

    signal += rng.normal(0, noise_std, n_samples)
    return signal


# ------------------------------------------------------------------
# Test 1: Feature extraction on synthetic signal (HR/RR)
# ------------------------------------------------------------------

def test_feature_extraction_known_rr() -> None:
    """Extract features from a synthetic ECG with known heart rate.

    Expected RR interval for 72 bpm = 833.3 ms.
    The extracted mean RR should be within 10% of this.
    """
    hr_bpm = 72.0
    expected_rr_ms = 60_000.0 / hr_bpm  # 833.33 ms

    signal = _make_synthetic_ecg(heart_rate_bpm=hr_bpm, duration_s=10.0)

    # Wrap as (12, n_samples) — replicate lead II across all leads
    waveform = np.tile(signal, (12, 1))

    features = extract_beat_features(waveform, fs=FS, lead_ii_index=1)

    print(f"\n  Synthetic ECG feature extraction:")
    print(f"    Expected HR:   {hr_bpm:.1f} bpm")
    print(f"    Extracted HR:  {features.heart_rate_bpm:.1f} bpm")
    print(f"    Expected RR:   {expected_rr_ms:.1f} ms")
    print(f"    Extracted RR:  {features.mean_rr_ms:.1f} ms")
    print(f"    QRS width:     {features.qrs_width_ms:.1f} ms")
    print(f"    R-peak amp:    {features.mean_r_amplitude_mv:.3f} mV")
    print(f"    R-peaks found: {features.n_rpeaks}")

    assert features.n_rpeaks >= 5, (
        f"Expected at least 5 R-peaks, got {features.n_rpeaks}"
    )
    # RR interval within 10% of expected
    rr_error_pct = abs(features.mean_rr_ms - expected_rr_ms) / expected_rr_ms
    assert rr_error_pct < 0.10, (
        f"RR interval off by {rr_error_pct:.1%}: "
        f"expected {expected_rr_ms:.1f} ms, got {features.mean_rr_ms:.1f} ms"
    )
    # HR within 10% of expected
    hr_error_pct = abs(features.heart_rate_bpm - hr_bpm) / hr_bpm
    assert hr_error_pct < 0.10, (
        f"HR off by {hr_error_pct:.1%}: "
        f"expected {hr_bpm:.1f} bpm, got {features.heart_rate_bpm:.1f} bpm"
    )


# ------------------------------------------------------------------
# Test 2: Feature extraction — QRS width in plausible range
# ------------------------------------------------------------------

def test_feature_extraction_qrs_width_plausible() -> None:
    """Extract QRS width from a synthetic ECG with a wider QRS complex.

    The synthetic signal is constructed with a ~50 ms QRS target.
    The tangent method on single-lead preprocessed signal should
    return a value in the plausible range (25-80 ms).
    """
    signal = _make_synthetic_ecg_wide_qrs(heart_rate_bpm=72.0, duration_s=10.0, qrs_width_ms=50.0)

    waveform = np.tile(signal, (12, 1))
    features = extract_beat_features(waveform, fs=FS, lead_ii_index=1)

    print(f"\n  Synthetic ECG (wide QRS) feature extraction:")
    print(f"    QRS width:     {features.qrs_width_ms:.1f} ms")
    print(f"    R-peaks found: {features.n_rpeaks}")

    assert features.n_rpeaks >= 5, (
        f"Expected at least 5 R-peaks, got {features.n_rpeaks}"
    )
    # QRS width in plausible single-lead range (25-80 ms)
    assert 25.0 <= features.qrs_width_ms <= 80.0, (
        f"QRS width {features.qrs_width_ms:.1f} ms outside plausible range 25-80 ms"
    )


# ------------------------------------------------------------------
# Test 3: Drift flag triggers on obvious shift
# ------------------------------------------------------------------

def test_drift_flag_triggers_on_shift() -> None:
    """Drift detection should flag an obvious heart-rate change.

    We construct BeatFeatures directly (bypassing R-peak detection on
    an impossibly-fast synthetic) to test that the z-score / flagging
    logic works correctly.

    Baseline: 72 bpm / 833 ms RR, Follow-up: 50 bpm / 1200 ms RR.
    HR z-score = (50-72)/72 ≈ -0.31, RR z-score = (1200-833)/833 ≈ 0.44.
    We also inject a large QRS width shift (80→250 ms → z=2.125)
    to exceed the threshold.
    """
    baseline_feat = BeatFeatures(
        mean_rr_ms=833.0,
        heart_rate_bpm=72.0,
        qrs_width_ms=80.0,
        mean_r_amplitude_mv=1.5,
        n_rpeaks=11,
        fs=FS,
    )
    # Dramatic QRS widening (simulating a bundle-branch block)
    followup_feat = BeatFeatures(
        mean_rr_ms=1200.0,
        heart_rate_bpm=50.0,
        qrs_width_ms=250.0,   # 212.5% of baseline → z = 2.125
        mean_r_amplitude_mv=1.5,
        n_rpeaks=8,
        fs=FS,
    )

    config = DriftConfig(z_threshold=2.0)
    z_scores = compute_z_scores(baseline_feat, followup_feat, config)
    max_abs_z = max(abs(z) for z in z_scores.values()) if z_scores else 0.0
    flagged = max_abs_z > config.z_threshold

    print(f"\n  Drift detection — SHIFT case:")
    print(f"    Baseline:  HR={baseline_feat.heart_rate_bpm:.1f} bpm, "
          f"RR={baseline_feat.mean_rr_ms:.1f} ms, QRS={baseline_feat.qrs_width_ms:.1f} ms")
    print(f"    Follow-up: HR={followup_feat.heart_rate_bpm:.1f} bpm, "
          f"RR={followup_feat.mean_rr_ms:.1f} ms, QRS={followup_feat.qrs_width_ms:.1f} ms")
    print(f"    Z-scores:     {z_scores}")
    print(f"    Max |z|:      {max_abs_z:.4f}")
    print(f"    Threshold:    {config.z_threshold}")
    print(f"    Drift flagged: {flagged}")

    assert flagged, (
        f"Drift should be flagged for QRS 80→250 ms shift, "
        f"but max |z|={max_abs_z:.4f} < threshold={config.z_threshold}"
    )


# ------------------------------------------------------------------
# Test 4: Drift flag does NOT trigger on same signal
# ------------------------------------------------------------------

def test_drift_flag_not_triggered_on_no_shift() -> None:
    """Drift detection should NOT flag when baseline and follow-up are similar.

    Both represent 72 bpm with only tiny feature differences.
    """
    baseline_feat = BeatFeatures(
        mean_rr_ms=833.0,
        heart_rate_bpm=72.0,
        qrs_width_ms=85.0,
        mean_r_amplitude_mv=1.5,
        n_rpeaks=11,
        fs=FS,
    )
    # Tiny variation — within 5% of baseline
    followup_feat = BeatFeatures(
        mean_rr_ms=840.0,
        heart_rate_bpm=71.4,
        qrs_width_ms=87.0,
        mean_r_amplitude_mv=1.49,
        n_rpeaks=11,
        fs=FS,
    )

    config = DriftConfig(z_threshold=2.0)
    z_scores = compute_z_scores(baseline_feat, followup_feat, config)
    max_abs_z = max(abs(z) for z in z_scores.values()) if z_scores else 0.0
    flagged = max_abs_z > config.z_threshold

    print(f"\n  Drift detection — NO SHIFT case:")
    print(f"    Baseline:  HR={baseline_feat.heart_rate_bpm:.1f} bpm, "
          f"RR={baseline_feat.mean_rr_ms:.1f} ms, QRS={baseline_feat.qrs_width_ms:.1f} ms")
    print(f"    Follow-up: HR={followup_feat.heart_rate_bpm:.1f} bpm, "
          f"RR={followup_feat.mean_rr_ms:.1f} ms, QRS={followup_feat.qrs_width_ms:.1f} ms")
    print(f"    Z-scores:     {z_scores}")
    print(f"    Max |z|:      {max_abs_z:.4f}")
    print(f"    Threshold:    {config.z_threshold}")
    print(f"    Drift flagged: {flagged}")

    assert not flagged, (
        f"Drift should NOT be flagged for same-HR signals, "
        f"but max |z|={max_abs_z:.4f} > threshold={config.z_threshold}"
    )
