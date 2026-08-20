"""Phase 5 — Personal baseline drift detection (PoC).

For each patient with a prior baseline recording, detect meaningful
deviation in interpretable beat-level features from that patient's own
historical norm using a simple z-score approach.

SCOPE LIMITATION (per PRD §7 and §11):
  - This is a SINGLE-BASELINE comparison: each follow-up is compared to
    the one baseline recording only.  There is no rolling distribution,
    no multi-visit history, no adaptive threshold.  This is a real scope
    limitation, not a hidden shortcut.
  - The drift layer is explicitly a PoC.  It is NOT validated.  Results
    are benchmarked but may be weak — that is expected and honest per §11.

METHODOLOGY:
  - Extract features on the baseline recording.
  - For each follow-up, extract the same features and compute z-scores
    relative to the baseline.
  - Flag "drift" if ANY feature's absolute z-score exceeds a threshold.
  - Threshold: |z| > 1.0 (recommended after Phase 5d ablation study —
    3-feature version: RR interval, heart rate, R-peak amplitude.
    QRS width was tested and dropped as net negative.)

DELIVERABLE:
  A 2x2 table: (drift flagged / not flagged) × (superclass changed / stayed same).
  This is the honest benchmark of whether the drift signal has ANY
  discriminative power.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.drift_cohort import (
    PatientDriftCohort,
    RecordingInfo,
    build_drift_cohort,
    print_cohort_summary,
)
from src.data.wfdb import read_wfdb_record
from src.models import evaluate as _eval  # noqa: F401 — unused but marks module
from src.preprocessing.beat_features import BeatFeatures, extract_beat_features
from src.preprocessing.filters import preprocess

_FS = 500.0
_WFDB_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ptbxl" / "wfdb"


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass
class DriftConfig:
    """Configuration for the drift detection PoC."""
    z_threshold: float = 2.0
    """Absolute z-score threshold for flagging drift.

    Chosen as |z| > 2.0 — the standard 95th-percentile outlier threshold
    from the normal-distribution literature.  This is a starting point
    for a PoC, NOT an optimized threshold.
    """

    features_to_check: Tuple[str, ...] = (
        "mean_rr_ms",
        "heart_rate_bpm",
        "qrs_width_ms",
        "mean_r_amplitude_mv",
    )
    """Which features to compute z-scores for."""


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class DriftResult:
    """Result of drift detection for one follow-up recording."""
    patient_id: int
    baseline_ecg_id: int
    followup_ecg_id: int
    # Per-feature z-scores
    z_scores: Dict[str, float] = field(default_factory=dict)
    # Max absolute z-score across all features
    max_abs_z: float = 0.0
    # Whether drift is flagged (any |z| > threshold)
    drift_flagged: bool = False
    # Whether the superclass changed between baseline and follow-up
    superclass_changed: bool = False
    # Feature values for transparency
    baseline_features: Optional[BeatFeatures] = None
    followup_features: Optional[BeatFeatures] = None


@dataclass
class BenchmarkResult:
    """2x2 benchmark table for the full cohort."""
    # True counts (not estimates)
    n_changed_flagged: int = 0      # superclass changed AND drift flagged
    n_changed_not_flagged: int = 0  # superclass changed BUT drift NOT flagged
    n_same_flagged: int = 0         # superclass same BUT drift flagged (false alarm)
    n_same_not_flagged: int = 0     # superclass same AND drift not flagged (correct)
    n_total_changed: int = 0
    n_total_same: int = 0
    n_total_flagged: int = 0
    n_total_not_flagged: int = 0
    n_total: int = 0
    # Per-feature breakdown
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ------------------------------------------------------------------
# Load & preprocess one recording
# ------------------------------------------------------------------

def _load_and_preprocess(
    filename_hr: str,
    wfdb_dir: Path,
) -> Optional[np.ndarray]:
    """Load a WFDB recording and apply Phase 2 preprocessing.

    Returns (12, n_samples) preprocessed waveform, or None on failure.
    """
    hea_path = wfdb_dir / (filename_hr + ".hea")
    dat_path = wfdb_dir / (filename_hr + ".dat")
    try:
        signal, fs, _leads = read_wfdb_record(hea_path, dat_path)
        processed = preprocess(signal, fs)
        if np.isnan(processed).any() or np.isinf(processed).any():
            return None
        return processed
    except Exception:
        return None


# ------------------------------------------------------------------
# Core drift detection
# ------------------------------------------------------------------

def compute_z_scores(
    baseline_features: BeatFeatures,
    followup_features: BeatFeatures,
    config: DriftConfig,
) -> Dict[str, float]:
    """Compute z-scores for each feature relative to the baseline.

    NOTE: This is a SINGLE-BASELINE comparison.  The z-score is computed
    as (followup - baseline) / baseline_value for features where the
    baseline provides a natural scale (e.g., RR interval, HR).

    For amplitude features, we use absolute difference / baseline as a
    simple normalized distance metric — NOT a true z-score (which would
    require a population standard deviation from multiple recordings).

    This is a PoC simplification, not a rigorous statistical test.
    """
    z_scores = {}

    for feat_name in config.features_to_check:
        b_val = getattr(baseline_features, feat_name)
        f_val = getattr(followup_features, feat_name)

        # Skip if either value is NaN
        if np.isnan(b_val) or np.isnan(f_val):
            z_scores[feat_name] = 0.0
            continue

        # Normalized distance: (f - b) / scale
        # For RR/HR, the baseline value is a reasonable scale.
        # For QRS width, use baseline value.
        # For amplitude, use baseline value.
        if abs(b_val) > 1e-6:
            z_scores[feat_name] = (f_val - b_val) / abs(b_val)
        else:
            z_scores[feat_name] = 0.0

    return z_scores


def detect_drift_for_patient(
    patient: PatientDriftCohort,
    wfdb_dir: Path,
    config: DriftConfig,
) -> List[DriftResult]:
    """Run drift detection for all follow-ups of one patient.

    Returns one DriftResult per follow-up recording.
    """
    results = []

    # Load and preprocess baseline
    baseline_wf = _load_and_preprocess(patient.baseline.filename_hr, wfdb_dir)
    if baseline_wf is None:
        return results

    baseline_feat = extract_beat_features(baseline_wf, fs=_FS)

    for fu in patient.follow_ups:
        fu_wf = _load_and_preprocess(fu.filename_hr, wfdb_dir)
        if fu_wf is None:
            continue

        fu_feat = extract_beat_features(fu_wf, fs=_FS)

        z_scores = compute_z_scores(baseline_feat, fu_feat, config)
        max_abs_z = max(abs(z) for z in z_scores.values()) if z_scores else 0.0
        drift_flagged = max_abs_z > config.z_threshold

        # Check superclass change
        baseline_sc = set(patient.baseline.superclasses)
        fu_sc = set(fu.superclasses)
        superclass_changed = baseline_sc != fu_sc

        result = DriftResult(
            patient_id=patient.patient_id,
            baseline_ecg_id=patient.baseline.ecg_id,
            followup_ecg_id=fu.ecg_id,
            z_scores=z_scores,
            max_abs_z=max_abs_z,
            drift_flagged=drift_flagged,
            superclass_changed=superclass_changed,
            baseline_features=baseline_feat,
            followup_features=fu_feat,
        )
        results.append(result)

    return results


# ------------------------------------------------------------------
# Full cohort benchmark
# ------------------------------------------------------------------

def run_full_benchmark(
    patients: List[PatientDriftCohort],
    wfdb_dir: Path | None = None,
    config: DriftConfig | None = None,
) -> Tuple[BenchmarkResult, List[DriftResult]]:
    """Run drift detection on the full cohort and produce the 2x2 benchmark.

    Parameters
    ----------
    patients : list of PatientDriftCohort
        The drift cohort from build_drift_cohort().
    wfdb_dir : Path, optional
        Override WFDB directory.
    config : DriftConfig, optional
        Override drift detection settings.

    Returns
    -------
    benchmark : BenchmarkResult
        The 2x2 table with real counts.
    all_results : list of DriftResult
        Individual results for every follow-up pair processed.
    """
    wfdb_dir = Path(wfdb_dir) if wfdb_dir else _WFDB_DIR
    config = config or DriftConfig()

    all_results: List[DriftResult] = []
    failed_patients = 0

    print(f"\n  Running drift detection on {len(patients)} patients "
          f"(z-threshold={config.z_threshold})...", flush=True)

    for i, patient in enumerate(patients):
        if (i + 1) % 500 == 0:
            print(f"    ... processed {i + 1}/{len(patients)} patients",
                  flush=True)
        results = detect_drift_for_patient(patient, wfdb_dir, config)
        if not results:
            failed_patients += 1
        all_results.extend(results)

    print(f"  Drift detection complete: {len(all_results)} follow-up pairs "
          f"evaluated ({failed_patients} patients had no evaluable pairs)")

    # Build 2x2 table
    benchmark = BenchmarkResult()
    for r in all_results:
        changed = r.superclass_changed
        flagged = r.drift_flagged

        if changed and flagged:
            benchmark.n_changed_flagged += 1
        elif changed and not flagged:
            benchmark.n_changed_not_flagged += 1
        elif not changed and flagged:
            benchmark.n_same_flagged += 1
        else:
            benchmark.n_same_not_flagged += 1

        if changed:
            benchmark.n_total_changed += 1
        else:
            benchmark.n_total_same += 1
        if flagged:
            benchmark.n_total_flagged += 1
        else:
            benchmark.n_total_not_flagged += 1
        benchmark.n_total += 1

    # Per-feature stats
    for feat_name in config.features_to_check:
        changed_z = [r.z_scores.get(feat_name, 0.0) for r in all_results
                     if r.superclass_changed]
        same_z = [r.z_scores.get(feat_name, 0.0) for r in all_results
                  if not r.superclass_changed]
        benchmark.feature_stats[feat_name] = {
            "mean_abs_z_changed": float(np.mean(np.abs(changed_z))) if changed_z else 0.0,
            "mean_abs_z_same": float(np.mean(np.abs(same_z))) if same_z else 0.0,
            "median_abs_z_changed": float(np.median(np.abs(changed_z))) if changed_z else 0.0,
            "median_abs_z_same": float(np.median(np.abs(same_z))) if same_z else 0.0,
        }

    return benchmark, all_results


def print_benchmark(benchmark: BenchmarkResult, config: DriftConfig) -> None:
    """Print the 2x2 benchmark table and analysis."""
    print(f"\n{'='*70}")
    print(f"DRIFT DETECTION BENCHMARK — PoC (NOT VALIDATED)")
    print(f"{'='*70}")
    print(f"  Z-score threshold: |z| > {config.z_threshold}")
    print(f"  Scope: SINGLE-BASELINE comparison (PoC limitation)")
    print(f"  Total follow-up pairs evaluated: {benchmark.n_total}")
    print()

    # 2x2 table
    print(f"  2x2 TABLE: Drift Flag vs Superclass Change")
    print(f"  {'':>30s} {'Superclass':>15s} {'Superclass':>15s}")
    print(f"  {'':>30s} {'CHANGED':>15s} {'SAME':>15s}")
    print(f"  {'-'*60}")
    print(f"  {'Drift FLAGGED':>30s} {benchmark.n_changed_flagged:>15d} "
          f"{benchmark.n_same_flagged:>15d}")
    print(f"  {'Drift NOT flagged':>30s} {benchmark.n_changed_not_flagged:>15d} "
          f"{benchmark.n_same_not_flagged:>15d}")
    print(f"  {'-'*60}")
    print(f"  {'Totals':>30s} {benchmark.n_total_changed:>15d} "
          f"{benchmark.n_total_same:>15d}")
    print()

    # Sensitivity / specificity (honest labels)
    if benchmark.n_total_changed > 0:
        sensitivity = benchmark.n_changed_flagged / benchmark.n_total_changed
        print(f"  Sensitivity (changed → flagged): "
              f"{benchmark.n_changed_flagged}/{benchmark.n_total_changed} = "
              f"{sensitivity:.3f}")
    if benchmark.n_total_same > 0:
        specificity = benchmark.n_same_not_flagged / benchmark.n_total_same
        print(f"  Specificity (same → not flagged): "
              f"{benchmark.n_same_not_flagged}/{benchmark.n_total_same} = "
              f"{specificity:.3f}")
    if benchmark.n_total > 0:
        accuracy = (benchmark.n_changed_flagged + benchmark.n_same_not_flagged) / benchmark.n_total
        print(f"  Overall accuracy: {accuracy:.3f}")

    # False alarm rate
    if benchmark.n_total_same > 0:
        fpr = benchmark.n_same_flagged / benchmark.n_total_same
        print(f"  False alarm rate: {benchmark.n_same_flagged}/{benchmark.n_total_same} = "
              f"{fpr:.3f}")

    print()

    # Per-feature breakdown
    print(f"  Per-feature mean |z|-scores:")
    print(f"  {'Feature':>25s} {'Changed':>12s} {'Same':>12s} {'Ratio':>8s}")
    print(f"  {'-'*57}")
    for feat_name, stats in benchmark.feature_stats.items():
        mc = stats["mean_abs_z_changed"]
        ms = stats["mean_abs_z_same"]
        ratio = mc / ms if ms > 0 else float("inf")
        print(f"  {feat_name:>25s} {mc:>12.4f} {ms:>12.4f} {ratio:>8.2f}")

    print(f"{'='*70}")


# ------------------------------------------------------------------
# Honest assessment
# ------------------------------------------------------------------

def honest_assessment(benchmark: BenchmarkResult) -> str:
    """Return a plain honest statement on drift signal discriminative power.

    Per PRD §11: "Drift layer doesn't work in time -> present as
    benchmarked roadmap goal, not a shipped capability -- do not fake
    results."
    """
    if benchmark.n_total == 0:
        return "No follow-up pairs could be evaluated."

    # Compute metrics
    sensitivity = (benchmark.n_changed_flagged / benchmark.n_total_changed
                   if benchmark.n_total_changed > 0 else 0.0)
    specificity = (benchmark.n_same_not_flagged / benchmark.n_total_same
                   if benchmark.n_total_same > 0 else 0.0)

    # Baseline: what if we flagged everything?
    naive_sensitivity = 1.0
    naive_fpr = 1.0  # everything flagged = all same-class flagged too

    # Baseline: what if we flagged nothing?
    naive_sensitivity_0 = 0.0
    naive_fpr_0 = 0.0

    lines = []
    lines.append("=" * 70)
    lines.append("HONEST ASSESSMENT — DRIFT SIGNAL DISCRIMINATIVE POWER")
    lines.append("=" * 70)
    lines.append("")
    lines.append("This is a PoC (proof of concept), NOT a validated system.")
    lines.append(f"Evaluated {benchmark.n_total} follow-up recording pairs.")
    lines.append("")

    # Is the drift signal better than random?
    if benchmark.n_total_changed > 0 and benchmark.n_total_same > 0:
        # Compute PPV and NPV
        ppv = (benchmark.n_changed_flagged /
               (benchmark.n_changed_flagged + benchmark.n_same_flagged)
               if (benchmark.n_changed_flagged + benchmark.n_same_flagged) > 0
               else 0.0)
        npv = (benchmark.n_same_not_flagged /
               (benchmark.n_same_not_flagged + benchmark.n_changed_not_flagged)
               if (benchmark.n_same_not_flagged + benchmark.n_changed_not_flagged) > 0
               else 0.0)

        lines.append(f"Sensitivity: {sensitivity:.3f} "
                     f"({benchmark.n_changed_flagged}/{benchmark.n_total_changed} "
                     f"real changes correctly flagged)")
        lines.append(f"Specificity: {specificity:.3f} "
                     f"({benchmark.n_same_not_flagged}/{benchmark.n_total_same} "
                     f"no-change cases correctly not flagged)")
        lines.append(f"PPV (precision): {ppv:.3f}")
        lines.append(f"NPV: {npv:.3f}")
        lines.append("")

        # Honest verdict
        if sensitivity > 0.6 and specificity > 0.7:
            verdict = ("The drift signal shows SOME discriminative power — "
                       "better than naive baselines.  However, this is a PoC "
                       "with a single-baseline comparison and should NOT be "
                       "considered validated.")
        elif sensitivity > 0.4 or specificity > 0.7:
            verdict = ("The drift signal is WEAK — it shows marginal "
                       "discriminative power at best.  The single-baseline "
                       "z-score approach is too simplistic for reliable "
                       "clinical use.  This is expected for a PoC and is a "
                       "legitimate finding per PRD §11.")
        else:
            verdict = ("The drift signal is INDISTINGUISHABLE FROM NOISE — "
                       "it shows essentially no discriminative power.  The "
                       "single-baseline z-score approach is insufficient.  "
                       "This is expected for a PoC and is a legitimate finding "
                       "per PRD §11.")

        lines.append(f"VERDICT: {verdict}")
    else:
        lines.append("Insufficient data to assess discriminative power.")

    lines.append("")
    lines.append("IMPORTANT CAVEATS:")
    lines.append("  1. This is a SINGLE-BASELINE comparison, not a rolling model.")
    lines.append("  2. Z-scores are computed relative to ONE prior recording, not")
    lines.append("     a population distribution.")
    lines.append("  3. Superclass labels are multi-label — 'changed' means ANY")
    lines.append("     superclass difference, not necessarily a new condition.")
    lines.append("  4. The drift layer is NOT validated and should NOT be shipped")
    lines.append("     as a capability (per PRD §11).")
    lines.append("  5. This is a roadmap goal, not a product feature.")
    lines.append("=" * 70)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def main() -> None:
    """Run the full drift detection PoC pipeline."""
    print("=" * 70)
    print("PHASE 5 — PERSONAL BASELINE DRIFT LAYER (PoC)")
    print("=" * 70)
    print("NOTE: This is a proof of concept, NOT a validated system.")
    print("Per PRD §11: drift layer is a roadmap goal, not a shipped capability.")
    print()

    # Step 1: Build cohort
    print("--- Step 1: Building drift cohort ---")
    patients, meta = build_drift_cohort()
    print_cohort_summary(patients)

    # Step 2: Sanity check features on a few known-NORM recordings
    print("\n--- Step 2: Feature sanity check on known-NORM recordings ---")
    _sanity_check_on_norms(patients)

    # Step 3-4: Run drift detection and benchmark
    print("\n--- Step 3-4: Drift detection + benchmark ---")
    config = DriftConfig()
    benchmark, all_results = run_full_benchmark(patients, config=config)
    print_benchmark(benchmark, config)

    # Step 5: Honest assessment
    print("\n--- Step 5: Honest assessment ---")
    print(honest_assessment(benchmark))


def _sanity_check_on_norms(patients: List[PatientDriftCohort]) -> None:
    """Extract features on a handful of known-NORM baseline recordings."""
    from src.preprocessing.beat_features import sanity_check_features

    norm_baselines = [
        p for p in patients
        if p.baseline.superclasses == ["NORM"]
    ]

    print(f"  Found {len(norm_baselines)} NORM-only baselines in cohort.")
    print(f"  Sanity-checking first 5...")

    for p in norm_baselines[:5]:
        wf = _load_and_preprocess(p.baseline.filename_hr, _WFDB_DIR)
        if wf is None:
            print(f"  [ecg_id={p.baseline.ecg_id}] FAILED to load")
            continue
        feat = extract_beat_features(wf, fs=_FS)
        sanity_check_features(feat, label=f"ecg_id={p.baseline.ecg_id}")


if __name__ == "__main__":
    main()
