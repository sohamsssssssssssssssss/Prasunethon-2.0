"""Phase 2 tests for the preprocessing pipeline.

Run:  .venv/bin/python -m pytest tests/test_preprocessing.py -v -s
(-s so the printed before/after numbers are visible)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.filters import preprocess  # noqa: E402

FS = 500.0
DUR = 10.0
T = np.arange(int(FS * DUR)) / FS


def _power_at(x: np.ndarray, fs: float, f0: float) -> float:
    """Integrated PSD power in a 2 Hz band centred on f0."""
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=min(4096, len(x)))
    mask = (freqs >= f0 - 1.0) & (freqs <= f0 + 1.0)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def test_preprocess_preserves_shape() -> None:
    """preprocess() output must have the same shape as its input."""
    # 1-D (single lead)
    x1 = np.sin(2 * np.pi * 1.0 * T)
    y1 = preprocess(x1, FS)
    assert y1.shape == x1.shape
    assert y1.ndim == 1
    assert np.all(np.isfinite(y1))

    # 2-D (12 leads x samples) — a realistic PTB-XL batch
    rng = np.random.default_rng(0)
    x2 = np.stack([np.sin(2 * np.pi * 1.0 * T) + 0.1 * rng.standard_normal(T.size)
                   for _ in range(12)])
    y2 = preprocess(x2, FS)
    assert y2.shape == x2.shape
    assert y2.ndim == 2
    assert np.all(np.isfinite(y2))

    # length must be identical (no trimming)
    assert y1.size == x1.size
    assert y2.shape[1] == x2.shape[1]


def test_synthetic_50hz_noise_power_reduced() -> None:
    """A pure 1 Hz ECG-ish tone + injected 50 Hz powerline noise must have
    50 Hz power reduced by preprocessing. Prints real before/after numbers."""
    x = np.sin(2 * np.pi * 1.0 * T) + 0.3 * np.sin(2 * np.pi * 50.0 * T)
    y = preprocess(x, FS)

    before = _power_at(x, FS, 50.0)
    after = _power_at(y, FS, 50.0)
    print(f"\n  50 Hz power: before={before:.3e}  after={after:.3e}  "
          f"reduction={(1 - after / before):.1%}")
    assert after < before * 0.05, (
        f"50 Hz power not reduced enough: {before:.3e} -> {after:.3e}"
    )


def test_synthetic_60hz_noise_power_reduced() -> None:
    """Same for 60 Hz (US powerline)."""
    x = np.sin(2 * np.pi * 1.0 * T) + 0.3 * np.sin(2 * np.pi * 60.0 * T)
    y = preprocess(x, FS)

    before = _power_at(x, FS, 60.0)
    after = _power_at(y, FS, 60.0)
    print(f"\n  60 Hz power: before={before:.3e}  after={after:.3e}  "
          f"reduction={(1 - after / before):.1%}")
    assert after < before * 0.05, (
        f"60 Hz power not reduced enough: {before:.3e} -> {after:.3e}"
    )


def test_signal_content_preserved() -> None:
    """The underlying 1 Hz signal must survive (its power should not collapse)."""
    x = np.sin(2 * np.pi * 1.0 * T) + 0.3 * np.sin(2 * np.pi * 50.0 * T)
    y = preprocess(x, FS)
    before = _power_at(x, FS, 1.0)
    after = _power_at(y, FS, 1.0)
    print(f"\n  1 Hz power: before={before:.3e}  after={after:.3e}  "
          f"retained={after / before:.1%}")
    assert after > before * 0.5, (
        f"1 Hz signal destroyed by preprocessing: {before:.3e} -> {after:.3e}"
    )
