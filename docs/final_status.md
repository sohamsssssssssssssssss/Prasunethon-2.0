# Cadence — Final Status (Phase 10)

**Date:** 2026-08-20
**State:** All phases 1–10 complete. Tests pass. Dashboard boots.

---

## What's Built

| Phase | What | Key result |
|---|---|---|
| 1 | Data pipeline + patient-independent split | 21,799 records, 18,869 patients, leak-free split verified |
| 2 | Preprocessing (0.5–40 Hz bandpass + db4 wavelet) | 94–97% noise reduction, 0.0ms R-peak timing shift |
| 3 | Multi-label classifier (88K-param 1D-CNN) | Macro-F1 0.723, micro-F1 0.752 on held-out test |
| 4 | Temperature calibration + reject policy | 92.1% accuracy on kept decisions, 76.3% coverage |
| 5/5d | Drift layer PoC + ablation | Sensitivity 15.3%, specificity 89.3% at |z|>1.0 (3 features) |
| 6/6b | Latency benchmark | 3.78ms mean end-to-end (CPU, Apple M5) |
| 7/7b | Streamlit dashboard | Live inference + static fallback + drift analysis |
| 8 | Interactive tradeoff sliders | Precomputed curves, no live recomputation |
| 9 | Honest-claims audit | 3 minor flags, no §9 language violations |
| 10 | Final wrap-up | Stale items fixed, unverified claims corrected |

---

## What's Honestly Weak

- **Drift PoC sensitivity: 15.3%** — misses ~85% of real superclass changes. This is a benchmarked PoC finding, not a clinically useful feature. The always-predict-same baseline gets 51.6% accuracy; the drift layer sits close to that floor.
- **QRS width extraction: dropped** — produced 30–40ms (vs. clinical 80–120ms) due to preprocessing attenuation. Ablation confirmed it was net negative (improved specificity/PPV when removed). No PTB-XL ground-truth QRS annotation exists to validate against.
- **Latency is software-simulated only** — real-time-paced playback from recorded signals, not actual hardware integration. Sub-millisecond p99 (0.87ms) but occasional OS-level scheduling spikes push individual chunks to 2–4ms (0.14% of chunks).
- **Single-baseline drift comparison** — each follow-up is compared to one prior recording only. No rolling distribution, no multi-visit history, no adaptive threshold.

---

## What's NOT Done (correctly out of scope per PRD §5)

- **No deck/slide file** — must be created and audited against §9 before submission
- **No demo script / talking-points file** — must be created and audited against §9 before submission
- **No real hardware integration** — software-only, simulated streaming
- **No Indian clinical validation** — tested on PTB-XL (German hospital data), not CHC/PHC environments
- **No regulatory submission** — research prototype, not a medical device

---

## How to Run

### Dashboard
```bash
streamlit run app.py
# Opens at http://localhost:8501
# Three tabs: Live Inference, Static Results, Try the Tradeoff
```

### Test suite
```bash
.venv/bin/python -m pytest tests/ -v
# 26 tests, all pass
```

### Full pipeline (data → model → evaluation)
```bash
# Phase 1: Data verification
.venv/bin/python tests/test_split_integrity.py

# Phase 2: Preprocessing QC
.venv/bin/python scripts/preprocessing_qc.py

# Phase 3-4: Training + calibration (already done — checkpoints in checkpoints/)
.venv/bin/python -m src.models.evaluate
.venv/bin/python -m src.models.reject --test

# Phase 5d: Drift ablation
.venv/bin/python scripts/drift_ablation.py --output-json notebooks/latency_qc/drift_sweep_3feature.json

# Phase 6: Latency benchmark
.venv/bin/python -m src.eval.latency_benchmark
```

---

## Key Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (live + static + tradeoff) |
| `checkpoints/best_model.pt` | Trained model weights |
| `checkpoints/calibration_temperature.json` | Temperature scaling factor (T=1.043) |
| `checkpoints/reject_policy.json` | Per-class reject thresholds + threshold-coverage curve |
| `notebooks/latency_qc/latency_benchmark.json` | Measured latency stats |
| `notebooks/latency_qc/drift_sweep_3feature.json` | 3-feature drift threshold sweep (validated) |
| `docs/drift_poc_summary.md` | Drift PoC honest summary |
| `docs/phase9_audit.md` | §9 honest-claims audit report |
| `docs/final_status.md` | This file |

---

## Honest Numbers (all re-verified Phase 10)

| Metric | Value | Source |
|---|---|---|
| Dataset | 21,799 records, 18,869 patients | ptbxl_database.csv |
| Test-set accuracy (micro-avg) | 85.6% | Re-computed from model + test set |
| Reject accuracy (threshold 0.75) | 92.1% | reject_policy.json |
| Coverage | 76.3% | reject_policy.json |
| Drift sensitivity (|z|>1.0) | 15.3% | drift_sweep_3feature.json |
| Drift specificity (|z|>1.0) | 89.3% | drift_sweep_3feature.json |
| Drift PPV (|z|>1.0) | 57.3% | drift_sweep_3feature.json |
| Latency mean (CPU) | 3.78ms | latency_benchmark.json |
| Latency p99 (CPU) | 4.35ms | latency_benchmark.json |
