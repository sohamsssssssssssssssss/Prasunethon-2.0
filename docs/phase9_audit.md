# Phase 9 — Honest-Claims Audit (PRD §9)

**Date:** 2026-08-20
**Scope:** Every submission-facing text artifact, checked against the four §9 rules.
**Status:** Audit pass — no code changes. Fixes listed as recommended follow-ups.

---

## §9 Rules (quoted for reference)

1. **No diagnostic language** — always "flags for review" / "triages" / "prioritizes"
2. **No clinical validation / regulatory claims** — never "clinically validated," "FDA," "CDSCO," "approved," "certified," or implied superiority to cleared devices
3. **No projected numbers presented as achieved** — mark projections explicitly
4. **Impact claims must be labeled projected** — cost/time/lives saved phrased as design goals unless directly measured

---

## Artifacts Checked

| # | Artifact | Location | Exists? | Claims-relevant? |
|---|---|---|---|---|
| 1 | README.md | `./README.md` | ✅ | ✅ — problem statement, status, warning |
| 2 | Drift PoC Summary | `docs/drift_poc_summary.md` | ✅ | ✅ — benchmark numbers, method, assessment |
| 3 | Dashboard disclaimer | `app.py` lines 49–55 | ✅ | ✅ — PRD §9 banner, always visible |
| 4 | Dashboard static fallback | `app.py` `render_static_fallback()` | ✅ | ✅ — Phase 4/5/6 numbers |
| 5 | Dashboard live inference labels | `app.py` live tab text | ✅ | ✅ — classification labels, drift section |
| 6 | Dashboard tradeoff tab | `app.py` `render_tradeoff_tab()` | ✅ | ✅ — slider plain-language sentences |
| 7 | drift.py docstring | `src/models/drift.py` lines 1–20 | ✅ | ⚠️ — threshold reference |
| 8 | reject.py docstring | `src/models/reject.py` line 1 | ✅ | ❌ — no claims |
| 9 | calibration.py docstring | `src/models/calibration.py` lines 1–2 | ✅ | ❌ — no claims |
| 10 | Deck/slide file | N/A | ❌ No `.pptx` in repo | ⚠️ — needs audit once written |
| 11 | Demo script / talking-points | N/A | ❌ None exists | ⚠️ — needs audit once written |
| 12 | PROMPT_LOG.md | `./PROMPT_LOG.md` | ✅ | ❌ — internal log, not submission-facing |

---

## Flagged Instances

### FLAG 1 — README.md: stale status section

**Location:** `README.md` lines 11–16

**Exact quote:**
```
- **Phase 1 (current):** repo scaffold + PTB-XL data verification — proving the
  patient-independent split is leak-free before any model code is written.
- **Phase 2 (planned):** bandpass filtering + wavelet denoising.
- **Phase 3+ (planned):** models and evaluation.
```

**Violated rule:** §9 rule 3 — presents Phase 1 as current when the project is at Phase 8b. Not a false-claim violation per se, but a stale-state misrepresentation that could cause a judge to underestimate scope.

**Proposed fix:** Update to reflect actual current state:
```markdown
## Status

- **Phase 1:** ✅ Repo scaffold + PTB-XL data verification — leak-free split confirmed.
- **Phase 2:** ✅ Bandpass filtering + wavelet denoising pipeline.
- **Phase 3:** ✅ Multi-label classifier trained and evaluated (macro-F1 0.723).
- **Phase 4:** ✅ Temperature calibration + confidence-reject policy (92.1% on kept, 76.3% coverage).
- **Phase 5:** ✅ Personal baseline drift layer PoC (benchmarked, not validated — per §11).
- **Phase 6:** ✅ Per-beat inference latency benchmark (3.78ms mean, CPU).
- **Phase 7:** ✅ Streamlit demo dashboard with live inference + static fallback.
- **Phase 8:** ✅ Interactive risk-threshold sliders (precomputed curves, no live recomputation).
```

---

### FLAG 2 — drift.py docstring: stale threshold reference

**Location:** `src/models/drift.py` line 19

**Exact quote:**
```
  - Threshold: |z| > 2.0 — a standard starting point from the outlier
```

**Violated rule:** §9 rule 3 (minor) — the docstring references the Phase 5 initial threshold of 2.0, not the recommended post-ablation threshold of 1.0. Not submission-facing (not shown in deck/dashboard), but misleading to anyone reading source code.

**Proposed fix:** Update to:
```
  - Threshold: |z| > 1.0 (recommended after Phase 5d ablation study).
```

---

### FLAG 3 — drift_poc_summary.md: "superclass changed" ambiguity

**Location:** `docs/drift_poc_summary.md` line 27

**Exact quote:**
```
ground truth = whether PTB-XL superclass label changed between baseline and follow-up
```

**Violated rule:** §9 rule 4 (borderline) — "ground truth" implies a definitive clinical truth, but PTB-XL superclass labels are the dataset's own annotations, not a clinical gold standard. This could be misread as a validation claim.

**Proposed fix:** Rephrase to:
```
ground truth = whether PTB-XL superclass annotation changed between baseline and follow-up
```

This is minor — "annotation" is more precise than "label" and avoids implying clinical ground truth.

---

## Non-Flagged Claims (verified correct)

### README.md
- ✅ `"Research-stage prototype. Not clinically validated. Do not use for medical diagnosis."` — correct, explicit §9 compliance
- ✅ `"patient-independent, stratified splits"` — accurate description of methodology
- ✅ `"leak-free evaluation"` — verified by test_split_integrity.py
- ✅ Warning section — clear, honest, appropriately scoped

### Dashboard disclaimer (app.py lines 49–55)
- ✅ `"triages and flags recordings for human review"` — §9 rule 1 compliant
- ✅ `"it does not diagnose"` — explicit negation
- ✅ `"not clinically validated"` — §9 rule 2 compliant
- ✅ `"not cleared by any regulatory body"` — §9 rule 2 compliant
- ✅ `"research-stage proof of concept"` — honest framing

### Dashboard live inference tab
- ✅ `"Calibrated Probabilities & Predictions"` — uses ML terminology, not diagnostic
- ✅ `"flagged as 'needs review'"` — §9 rule 1 compliant
- ✅ `"should be reviewed by a clinician"` — defers to human judgment

### Dashboard static fallback
- ✅ `"Test-set accuracy (micro-avg) 85.6%"` — measured value, re-verified: actual 85.63%
- ~~`"After calibration 87.3%"`~~ — **REMOVED in Phase 10**: unverified figure, not traceable to any saved file. Temperature scaling does not change hard predictions.
- ✅ `"After reject (threshold 0.75) 92.1%"` — measured value, verified: actual 92.1%
- ✅ `"Coverage 76.3%"` — measured value, verified: actual 76.3%
- ✅ `"Drift Layer (PoC)"` — correctly labeled as PoC
- ✅ `"Real but weak signal — PoC, not validated"` — honest assessment

### Dashboard drift section
- ✅ `"proof-of-concept drift layer, NOT a validated feature"` — §9 rule 2 compliant
- ✅ `"benchmark numbers are real and unfiltered"` — §9 rule 3 compliant

### Dashboard tradeoff tab
- ✅ `"precomputed, validated table values — no metrics are recomputed at runtime"` — §9 rule 3 compliant
- ✅ `"Snaps to real precomputed threshold values only — no interpolation"` — honest
- ✅ Drift disclaimer: `"This is a benchmarked PoC finding, not a clinically useful triage feature"` — §9 rule 1 compliant

### docs/drift_poc_summary.md
- ✅ `"PoC — not validated, not a shipped capability"` — §9 rule 2 compliant
- ✅ `"explicitly scoped and reported as a proof-of-concept per the PRD's honesty boundary (§9, §11)"` — self-aware
- ✅ `"2 independent signals — RR and HR are mathematically linked"` — honest redundancy disclosure
- ✅ `"QRS width was tested and dropped"` — transparent ablation reporting
- ✅ `"the signal is real but weak"` — honest assessment
- ✅ `"always-predict-same classifier gets ~52% accuracy for free"` — verified: actual 51.6%, "~52%" is fair rounding
- ✅ `"not a clinically useful or validated triage feature"` — §9 rule 1 compliant

---

## Numeric Cross-Check

All numbers verified against source-of-truth data files:

| Claim | Source file | Actual value | Match? |
|---|---|---|---|
| 21,799 records | ptbxl_database.csv | 21,799 | ✅ |
| 18,869 patients | ptbxl_database.csv | 18,869 | ✅ |
| Baseline accuracy 85.8% | reject_policy.json | 85.84% | ✅ |
| Reject accuracy 92.1% | reject_policy.json (@0.75) | 92.08% | ✅ |
| Coverage 76.3% | reject_policy.json (@0.75) | 76.30% | ✅ |
| Drift cohort 2,109 patients | drift_sweep_3feature.json | 2,109 | ✅ |
| Drift follow-ups 2,926 | drift_sweep_3feature.json | 2,926 | ✅ |
| Sensitivity 15.3% | drift_sweep_3feature.json (@1.0) | 15.3% | ✅ |
| Specificity 89.3% | drift_sweep_3feature.json (@1.0) | 89.3% | ✅ |
| PPV 57.3% | drift_sweep_3feature.json (@1.0) | 57.3% | ✅ |
| Accuracy 53.5% | drift_sweep_3feature.json (@1.0) | 53.5% | ✅ |
| Always-predict-same ~52% | computed: 1509/2926 | 51.6% | ✅ |
| Latency mean 3.78ms | latency_benchmark.json | 3.78ms | ✅ |
| Latency p99 4.35ms | latency_benchmark.json | 4.35ms | ✅ |
| Hardware Apple M5 | latency_benchmark.json | Apple M5 | ✅ |

**No inflated, rounded-up, or projected-as-achieved numbers found.**

---

## Per-Artifact Verdict

| Artifact | Verdict | Notes |
|---|---|---|
| README.md | ⚠️ **CONDITIONAL CLEAR** | Stale status section needs update (Flag 1). No §9 language violations. |
| docs/drift_poc_summary.md | ✅ **CLEAR** | Minor "label"→"annotation" fix recommended (Flag 3). All numbers verified. |
| Dashboard disclaimer | ✅ **CLEAR** | Explicitly §9-compliant. Never softened. |
| Dashboard static fallback | ✅ **CLEAR** | All numbers match source files. |
| Dashboard live inference | ✅ **CLEAR** | Uses ML terminology, not diagnostic. |
| Dashboard tradeoff tab | ✅ **CLEAR** | Precomputed-only, no live recomputation. |
| drift.py docstring | ⚠️ **CONDITIONAL CLEAR** | Stale threshold (Flag 2). Not submission-facing. |
| reject.py docstring | ✅ **CLEAR** | No claims. |
| calibration.py docstring | ✅ **CLEAR** | No claims. |
| Deck file | ❌ **NOT YET CHECKED** | No `.pptx` exists in repo. Must audit once written. |
| Demo script | ❌ **NOT YET CHECKED** | No demo script exists. Must audit once written. |

---

## Overall Verdict

**CONDITIONAL CLEAR** — all existing submission-facing artifacts pass §9 audit with 3 minor flags (stale status, stale docstring threshold, "label" vs "annotation"). No diagnostic language, no validation claims, no inflated numbers, no unmarked projections found.

**Before submission, the following must also be audited:**
- [ ] Deck/slide file (if created) — apply same §9 checklist
- [ ] Demo script/talking-points (if created) — apply same §9 checklist
- [ ] README.md status section — update to reflect Phases 1–8b
- [ ] drift.py docstring — update threshold from 2.0 to 1.0
