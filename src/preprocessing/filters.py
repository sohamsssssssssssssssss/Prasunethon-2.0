"""Phase 2 preprocessing: bandpass filtering + wavelet denoising.

Design choices (with sources):

1. Bandpass 0.5-40 Hz, 4th-order Butterworth, zero-phase (filtfilt).
   - Baseline wander in ECG is below ~0.5 Hz (Kligfield et al., "Recommendations
     for the Standardization and Interpretation of the Electrocardiogram",
     Circulation 2007 — low-frequency noise / baseline wander < 0.5 Hz).
   - High-frequency muscle noise and the bulk of powerline harmonics live above
     40 Hz; the clinically relevant QRS/ST-T spectrum is below ~40 Hz.
   - 0.5-40 Hz 4th-order Butterworth is the standard preprocessing choice in
     the ECG literature (e.g., "Filtering of ECG signal using Butterworth
     Filter", IEEE 2014; used by PTB-XL downstream work; also the PhysioNet
     2021 Challenge baselines filter to ~0.5-100 Hz — we use 40 Hz since PTB-XL
     is a resting 12-lead recording and the 100 Hz range adds only noise).
   - filtfilt (zero-phase) avoids the phase distortion that would shift
     R-peak timing — critical for the Phase 3 timing-sensitive features.

2. Wavelet denoising: Daubechies-4 (db4), 4-level DWT, soft thresholding.
   - db4 is the classic ECG-denoising wavelet: compact support well matched to
     QRS morphology, and the de-facto standard in the ECG wavelet literature
     (e.g., Alfaouri & Daqrouq, "ECG Signal Denoising by Wavelet Transform
     Thresholding", 2008 — 5-level db4; also Addison, "Wavelet transforms and
     the ECG: a review", Physiol. Meas. 2005).
   - 4 levels at fs=500 Hz: detail bands ≈ [15.6-31.25, 31.25-62.5, 62.5-125,
     125-250] Hz. Muscle noise concentrates in the upper detail bands; the
     QRS/ST-T energy (mostly < 40 Hz) is preserved in the approximation + lower
     details. Soft thresholding (universal threshold, scipy default) removes
     small coefficients (noise) while keeping large ones (signal).

3. preprocess() chains bandpass -> wavelet denoise, identically at train and
   inference time.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

# Filter design parameters (see module docstring for rationale/sources).
BANDPASS_LOW = 0.5  # Hz — removes baseline wander (< 0.5 Hz)
BANDPASS_HIGH = 40.0  # Hz — removes muscle noise / powerline harmonics
BANDPASS_ORDER = 4

# Wavelet parameters.
WAVELET = "db4"
DECOMPOSITION_LEVEL = 4


PAD_SECONDS = 6.0  # edge padding to kill filtfilt low-edge transients


def bandpass_filter(
    x: np.ndarray, fs: float, low: float = BANDPASS_LOW,
    high: float = BANDPASS_HIGH, order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass, applied along the last axis.

    The signal is padded with its edge values (PAD_SECONDS on each side)
    before filtering and trimmed after. A 0.5 Hz low edge at fs=500 has a long
    impulse response and filtfilt's default padding leaves a transient of up
    to ~2 s (visible as spurious sub-0.5 Hz power at the signal edges); the
    explicit edge padding removes it.
    """
    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid band [{low}, {high}] for fs={fs} (nyquist={nyq})")
    sos = sp_signal.butter(order, [low, high], btype="bandpass", fs=fs,
                           output="sos")

    x = np.asarray(x, dtype=np.float64)
    n_pad = int(PAD_SECONDS * fs)
    xp = np.concatenate(
        [np.broadcast_to(x[..., :1], (*x.shape[:-1], n_pad)), x,
         np.broadcast_to(x[..., -1:], (*x.shape[:-1], n_pad))],
        axis=-1,
    )
    yp = sp_signal.sosfiltfilt(sos, xp, axis=-1)
    return yp[..., n_pad:-n_pad]


def wavelet_denoise(
    x: np.ndarray, wavelet: str = WAVELET, level: int = DECOMPOSITION_LEVEL,
) -> np.ndarray:
    """DWT + soft-threshold denoise, applied along the last axis."""
    import pywt

    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y[None, :]
    out = np.empty_like(y)
    for i, row in enumerate(y):
        coeffs = pywt.wavedec(row, wavelet, level=level)
        # Universal (VisuShrink) threshold, estimated from the finest detail.
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thresh = sigma * np.sqrt(2.0 * np.log(len(row)))
        coeffs = [coeffs[0]] + [
            pywt.threshold(c, thresh, mode="soft") for c in coeffs[1:]
        ]
        out[i] = pywt.waverec(coeffs, wavelet)[: len(row)]
    return out if x.ndim > 1 else out[0]


def preprocess(raw_signal: np.ndarray, fs: float) -> np.ndarray:
    """Chain bandpass -> wavelet denoise. Same call at train and inference."""
    bandpassed = bandpass_filter(raw_signal, fs=fs)
    return wavelet_denoise(bandpassed)
