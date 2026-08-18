"""Phase 2 regression check: does preprocessing distort QRS timing?

On clean, noise-free PTB-XL recordings (no noise flags), detect R-peaks on
the RAW and PREPROCESSED lead II with the *same* detector and compare peak
locations. Reports the max timing shift in ms and in samples.

A zero-phase filter (filtfilt) should produce 0-sample shifts; the wavelet
step is approximately shift-preserving. Any shift > ~2 ms (1 sample at
500 Hz) would indicate over-smoothing that moves R-peaks.

Usage:  .venv/bin/python scripts/regression_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wfdb import read_wfdb_record  # noqa: E402
from src.preprocessing.filters import preprocess  # noqa: E402

FS = 500.0
# Clean NORM recordings with no noise flags at all.
CLEAN_RECS = [2, 3, 7]


def detect_rpeaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Simple QRS detector: 5-20 Hz bandpass, square, smooth, find_peaks.

    Identical preprocessing for raw and processed inputs so any timing
    difference is attributable to preprocess() itself.
    """
    sos = sp_signal.butter(4, [5.0, 20.0], btype="bandpass", fs=fs,
                           output="sos")
    bp = sp_signal.sosfiltfilt(sos, x)
    env = bp ** 2
    win = int(0.08 * fs)  # 80 ms moving average
    kernel = np.ones(win) / win
    env = np.convolve(env, kernel, mode="same")
    peaks, _ = sp_signal.find_peaks(env, distance=0.3 * fs,
                                    prominence=np.percentile(env, 95) * 0.3)
    return peaks


def main() -> None:
    print(f"{'ecg_id':>6s} {'lead':>4s} {'#peaks':>7s} {'#matched':>9s} "
          f"{'max|shift|(samp)':>15s} {'max|shift|(ms)':>15s}")
    worst_ms = 0.0
    for ecg_id in CLEAN_RECS:
        hea = f"data/ptbxl/wfdb/{ecg_id:05d}_hr.hea"
        sig, fs, leads = read_wfdb_record(hea)
        assert fs == FS
        lead_idx = leads.index("II")
        raw = sig[lead_idx]
        cleaned = preprocess(sig, fs)[lead_idx]

        p_raw = detect_rpeaks(raw, fs)
        p_cln = detect_rpeaks(cleaned, fs)
        assert len(p_raw) == len(p_cln), (
            f"ecg {ecg_id}: peak count differs raw={len(p_raw)} "
            f"processed={len(p_cln)}"
        )
        shifts = np.abs(p_raw - p_cln)
        max_samp = int(shifts.max()) if len(shifts) else 0
        max_ms = max_samp * 1000.0 / fs
        worst_ms = max(worst_ms, max_ms)
        print(f"{ecg_id:>6d} {'II':>4s} {len(p_raw):>7d} {len(p_cln):>9d} "
              f"{max_samp:>15d} {max_ms:>15.1f}")
        if len(p_raw):
            print(f"      R-peak times (s): raw     = "
                  f"{np.round(p_raw / fs, 2)}")
            print(f"                        processed = "
                  f"{np.round(p_cln / fs, 2)}")
    print()
    print(f"WORST-CASE R-PEAK TIMING SHIFT: {worst_ms:.1f} ms "
          f"({worst_ms * FS / 1000.0:.0f} samples at {FS:.0f} Hz)")


if __name__ == "__main__":
    main()
