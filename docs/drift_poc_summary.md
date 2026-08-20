## Personal Baseline Drift Layer (Proof-of-Concept)

**Status: PoC — not validated, not a shipped capability.**

Compares a patient's follow-up ECG against their own prior baseline
recording, rather than population norms alone. This is the project's
core differentiator, and it is explicitly scoped and reported as a
proof-of-concept per the PRD's honesty boundary (§9, §11) — the
numbers below are real and unfiltered, not cherry-picked.

**Features used:** RR interval, heart rate, R-peak amplitude
(2 independent signals — RR and HR are mathematically linked, so this
is not 3 independent measurements).

**QRS width was tested and dropped.** Initial extraction produced
30-40ms (vs. clinical 80-120ms). Root-cause investigation found a
double-filtering bug (fixed), but the corrected value still sits
below the literature-expected single-lead range (~60-90ms), and
PTB-XL provides no ground-truth QRS annotation to validate against.
An ablation study (dropping QRS width entirely) showed specificity
and PPV improved at every threshold tested, with only a 1-4.5pp
sensitivity cost — QRS width was net negative, not just unvalidated.
It was removed from the final feature set on that basis.

**Method:** single-baseline z-score `(follow-up − baseline) / |baseline|`
per feature, threshold swept across |z| ∈ [0.25, 3.0]. Not a
population-normalized z-score — this is a known scope limitation of
the single-baseline PoC design, not a hidden shortcut.

**Benchmark (2,109 patients, 2,926 follow-up recordings, ground truth
= whether PTB-XL superclass label changed between baseline and
follow-up):**

| Threshold | Sensitivity | Specificity | PPV | Accuracy |
|---|---|---|---|---|
| \|z\| > 1.0 (recommended) | 15.3% | 89.3% | 57.3% | 53.5% |

**Honest assessment:** the signal is real but weak. Base rate of
"superclass changed" in this cohort is 48.4%, meaning a trivial
always-predict-same classifier gets ~52% accuracy for free — the
drift layer sits close to that floor. The ablation confirms the
remaining 3-feature signal is genuine (consistent, one-directional
improvement over the 4-feature version across all thresholds
tested), but this is a benchmarked PoC finding, not a clinically
useful or validated triage feature. Presented in the deck/demo as a
roadmap direction, not a shipped capability, per PRD §11.
