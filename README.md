# Cadence

**Research-stage prototype. Not clinically validated. Do not use for medical diagnosis.**

## Problem statement

Cadence is a research project exploring patient-independent ECG interpretation
models trained on the PTB-XL dataset (Wagner et al., 2020, Scientific Data,
DOI 10.1038/s41597-020-0495-6). The goal is to evaluate whether
patient-independent, stratified splits of PTB-XL enable leak-free evaluation of
diagnostic superclass classification (NORM, MI, STTC, CD, HYP).

Full product requirements: see the PRD (linked from the project brief / prompt log).

## Status

All phases through 8b are complete and tested.

- **Phase 1:** ✅ Repo scaffold + PTB-XL data verification — leak-free patient-independent split confirmed.
- **Phase 2:** ✅ Bandpass filtering (0.5–40 Hz) + wavelet denoising (db4, level 4) pipeline.
- **Phase 3:** ✅ Multi-label classifier trained (88K-param 1D-CNN), evaluated on held-out test set (macro-F1 0.723).
- **Phase 4:** ✅ Temperature scaling + confidence-reject policy (90.4% accuracy on kept decisions, 82.3% coverage at threshold 0.70; see checkpoints/test_results_0.70_corrected.json).
- **Phase 5:** ✅ Personal baseline drift layer PoC — benchmarked but not validated (sensitivity 15.3%, specificity 89.3% at |z|>1.0). QRS width tested and dropped after ablation.
- **Phase 6:** ✅ Per-beat inference latency benchmark (3.78ms mean end-to-end on CPU, Apple M5).
- **Phase 7:** ✅ Streamlit demo dashboard with live inference, static fallback, and interactive tradeoff sliders.
- **Phase 8:** ✅ Interactive risk-threshold sliders using precomputed validated curves.
- **Phase 9:** ✅ Honest-claims audit against PRD §9 — 3 minor flags found, none are language violations.

**Known limitations:** drift PoC sensitivity is weak (15.3%), QRS width extraction was unvalidated and dropped, latency is software-simulated streaming only (not real hardware).

## Layout

```
cadence/
  data/               (gitignored — raw PTB-XL)
  src/data/           (loading, split logic)
  src/preprocessing/  (phase 2)
  src/models/         (phase 3+)
  notebooks/          (exploration only)
  tests/              (split-integrity verification)
```

## Warning

This project is a research prototype. It is not a medical device and has not
been clinically validated. It must not be used to inform clinical decisions.
# Prasunethon-2.0
# cadence
