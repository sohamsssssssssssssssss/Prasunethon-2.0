"""Phase 2 QC: before/after evidence for the preprocessing pipeline.

For each flagged-noisy PTB-XL recording:
  1. Loads the 500 Hz WFDB record, picks the flagged lead (or lead II).
  2. Applies preprocess() (bandpass 0.5-40 Hz + db4 wavelet denoise).
  3. Saves a raw-vs-preprocessed overlay PNG to notebooks/preprocessing_qc/.
  4. Prints quantitative noise-reduction numbers: power in the
     high-frequency muscle-noise band (40-250 Hz) and in the baseline-wander
     band (< 0.5 Hz) before vs after, plus an SNR estimate.

Usage:  .venv/bin/python scripts/preprocessing_qc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal as sp_signal  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.wfdb import read_wfdb_record  # noqa: E402
from src.preprocessing.filters import preprocess  # noqa: E402

FS = 500.0
OUT_DIR = Path("notebooks/preprocessing_qc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ecg_id -> (flag text, lead index to plot; lead II = index 1 if flag = all)
NOISY_RECS = [
    (38, "static_noise: 'alles' (all leads)", 1),     # lead II
    (80, "baseline_drift: 'alles' (all leads)", 1),   # lead II
    (100, "burst_noise: I-V1", 0),                    # lead I
    (128, "static + baseline drift: 'alles'", 1),     # lead II
    (133, "burst_noise: I,III,AVL", 0),               # lead I
]


def band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Integrated PSD power (mV^2) of x within [lo, hi] Hz via Welch."""
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (freqs >= lo) & (freqs <= hi)
    return float(np.trapezoid(psd[mask], freqs[mask]))


def snr_estimate(x: np.ndarray, fs: float) -> float:
    """SNR estimate: signal-band power (0.5-40 Hz) / noise-band power (40-250 Hz)."""
    sig_p = band_power(x, fs, 0.5, 40.0)
    noise_p = band_power(x, fs, 40.0, 250.0)
    if noise_p <= 0:
        return float("inf")
    return sig_p / noise_p


def main() -> None:
    print(f"{'ecg_id':>6s} {'lead':>4s} {'band':>14s} {'before(mV^2)':>13s} "
          f"{'after(mV^2)':>12s} {'reduction':>10s}")
    for ecg_id, flag_text, lead_idx in NOISY_RECS:
        hea = f"data/ptbxl/wfdb/{ecg_id:05d}_hr.hea"
        sig, fs, leads = read_wfdb_record(hea)
        assert fs == FS
        lead_name = leads[lead_idx]
        raw = sig[lead_idx]
        cleaned = preprocess(sig, fs)[lead_idx]

        b_norm = snr_estimate(raw, fs)
        b_clean = snr_estimate(cleaned, fs)

        for label, lo, hi in [("noise 40-250Hz", 40.0, 250.0),
                              ("baseline <0.5Hz", 0.0, 0.5)]:
            before = band_power(raw, fs, lo, hi)
            after = band_power(cleaned, fs, lo, hi)
            red = 1.0 - after / before if before > 0 else float("nan")
            print(f"{ecg_id:>6d} {lead_name:>4s} {label:>14s} "
                  f"{before:>13.2e} {after:>12.2e} {red:>9.1%}")
        print(f"      SNR(0.5-40/40-250): {b_norm:8.2f} -> {b_clean:8.2f} "
              f"({flag_text})")
        print()

        t = np.arange(len(raw)) / fs
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, raw, color="0.55", lw=0.8, label="raw")
        axes[0].set_title(f"ecg {ecg_id} ({lead_name}) — raw vs preprocessed")
        axes[0].set_ylabel("mV"); axes[0].legend(loc="upper right")
        axes[1].plot(t, raw, color="0.55", lw=0.8, label="raw")
        axes[1].plot(t, cleaned, color="crimson", lw=0.8, label="preprocessed")
        axes[1].set_ylabel("mV"); axes[1].set_xlabel("time (s)")
        axes[1].legend(loc="upper right")
        axes[0].set_xlim(0, 10)
        out = OUT_DIR / f"ecg_{ecg_id:05d}_qc.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
