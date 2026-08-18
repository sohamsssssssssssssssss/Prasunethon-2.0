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

- **Phase 1 (current):** repo scaffold + PTB-XL data verification — proving the
  patient-independent split is leak-free before any model code is written.
- **Phase 2 (planned):** bandpass filtering + wavelet denoising.
- **Phase 3+ (planned):** models and evaluation.

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
