"""Phase 5d — Ablation: drift detection without QRS width.

Re-runs drift detection on the same 2,109-patient / 2,926-follow-up cohort
using only 3 features (RR, HR, R-peak amplitude) — QRS width removed entirely
from the feature vector, not zero-weighted.

Efficiency: features are extracted ONCE per recording, then both the 4-feature
and 3-feature configs are evaluated on the cached features.  No double I/O.

Usage:  .venv/bin/python scripts/drift_ablation.py
       .venv/bin/python scripts/drift_ablation.py --output-json path/to/file.json
"""

from __future__ import annotations

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.drift_cohort import (
    PatientDriftCohort,
    build_drift_cohort,
    print_cohort_summary,
)
from src.data.wfdb import read_wfdb_record
from src.models.drift import (
    BenchmarkResult,
    DriftConfig,
    DriftResult,
    _WFDB_DIR,
    _FS,
    _load_and_preprocess,
    compute_z_scores,
)
from src.preprocessing.beat_features import BeatFeatures, extract_beat_features


# ------------------------------------------------------------------
# Feature extraction (run once, cache results)
# ------------------------------------------------------------------

def _extract_all_features(
    patients: List[PatientDriftCohort],
    wfdb_dir: Path,
) -> Dict[int, BeatFeatures]:
    """Extract BeatFeatures for every recording in the cohort.

    Returns a dict mapping ecg_id -> BeatFeatures.
    This avoids re-reading and re-preprocessing waveforms when we
    evaluate different feature subsets.
    """
    # Collect all unique ecg_ids we need
    ecg_ids_needed = set()
    for p in patients:
        ecg_ids_needed.add(p.baseline.ecg_id)
        for fu in p.follow_ups:
            ecg_ids_needed.add(fu.ecg_id)

    print(f"  Extracting features for {len(ecg_ids_needed)} unique recordings...",
          flush=True)

    # Build a lookup from ecg_id -> filename_hr
    ecg_to_file = {}
    for p in patients:
        ecg_to_file[p.baseline.ecg_id] = p.baseline.filename_hr
        for fu in p.follow_ups:
            ecg_to_file[fu.ecg_id] = fu.filename_hr

    features_cache: Dict[int, BeatFeatures] = {}
    n_ok = 0
    n_fail = 0

    for i, ecg_id in enumerate(sorted(ecg_ids_needed)):
        if (i + 1) % 500 == 0:
            print(f"    ... extracted {i + 1}/{len(ecg_ids_needed)} "
                  f"({n_ok} ok, {n_fail} failed)", flush=True)

        fname = ecg_to_file[ecg_id]
        wf = _load_and_preprocess(fname, wfdb_dir)
        if wf is None:
            n_fail += 1
            continue

        feat = extract_beat_features(wf, fs=_FS)
        features_cache[ecg_id] = feat
        n_ok += 1

    print(f"  Feature extraction complete: {n_ok} ok, {n_fail} failed "
          f"(out of {len(ecg_ids_needed)} recordings)")
    return features_cache


# ------------------------------------------------------------------
# Benchmark from cached features (no I/O)
# ------------------------------------------------------------------

def _benchmark_from_cache(
    patients: List[PatientDriftCohort],
    features_cache: Dict[int, BeatFeatures],
    config: DriftConfig,
) -> Tuple[BenchmarkResult, List[DriftResult]]:
    """Run drift detection using cached features — pure CPU, no disk I/O."""
    all_results: List[DriftResult] = []
    n_no_baseline = 0
    n_no_followup = 0

    for patient in patients:
        baseline_feat = features_cache.get(patient.baseline.ecg_id)
        if baseline_feat is None:
            n_no_baseline += 1
            continue

        for fu in patient.follow_ups:
            fu_feat = features_cache.get(fu.ecg_id)
            if fu_feat is None:
                n_no_followup += 1
                continue

            z_scores = compute_z_scores(baseline_feat, fu_feat, config)
            max_abs_z = max(abs(z) for z in z_scores.values()) if z_scores else 0.0
            drift_flagged = max_abs_z > config.z_threshold

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
            all_results.append(result)

    if n_no_baseline > 0:
        print(f"  WARNING: {n_no_baseline} patients had no extractable baseline")
    if n_no_followup > 0:
        print(f"  WARNING: {n_no_followup} follow-ups had no extractable features")

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


# ------------------------------------------------------------------
# Threshold sweep
# ------------------------------------------------------------------

def _compute_metrics(bm: BenchmarkResult) -> Dict[str, float]:
    """Extract key metrics from a BenchmarkResult."""
    sens = (bm.n_changed_flagged / bm.n_total_changed
            if bm.n_total_changed > 0 else 0.0)
    spec = (bm.n_same_not_flagged / bm.n_total_same
            if bm.n_total_same > 0 else 0.0)
    far = (bm.n_same_flagged / bm.n_total_same
           if bm.n_total_same > 0 else 0.0)
    ppv = (bm.n_changed_flagged /
           (bm.n_changed_flagged + bm.n_same_flagged)
           if (bm.n_changed_flagged + bm.n_same_flagged) > 0 else 0.0)
    acc = ((bm.n_changed_flagged + bm.n_same_not_flagged) / bm.n_total
           if bm.n_total > 0 else 0.0)
    return {
        "sensitivity": sens,
        "specificity": spec,
        "FAR": far,
        "PPV": ppv,
        "accuracy": acc,
        "flagged": bm.n_changed_flagged + bm.n_same_flagged,
    }


def threshold_sweep(
    patients: List[PatientDriftCohort],
    features_cache: Dict[int, BeatFeatures],
    feature_configs: Dict[str, Tuple[str, ...]],
    thresholds: List[float],
) -> Dict[str, List[Dict[str, float]]]:
    """Run threshold sweep for each feature configuration.

    Parameters
    ----------
    patients : the drift cohort
    features_cache : pre-extracted features (ecg_id -> BeatFeatures)
    feature_configs : {"4-feat": (rr, hr, qrs, amp), "3-feat": (rr, hr, amp)}
    thresholds : list of |z| thresholds to evaluate

    Returns
    -------
    Dict mapping config_name -> list of metric dicts (one per threshold)
    """
    results: Dict[str, List[Dict[str, float]]] = {}

    for config_name, feat_tuple in feature_configs.items():
        print(f"\n  === Threshold sweep: {config_name} ===")
        sweep_rows = []
        for thresh in thresholds:
            cfg = DriftConfig(z_threshold=thresh, features_to_check=feat_tuple)
            bm, _ = _benchmark_from_cache(patients, features_cache, cfg)
            metrics = _compute_metrics(bm)
            metrics["threshold"] = thresh
            metrics["n_changed"] = bm.n_total_changed
            metrics["n_same"] = bm.n_total_same
            metrics["n_changed_flagged"] = bm.n_changed_flagged
            metrics["n_same_not_flagged"] = bm.n_same_not_flagged
            sweep_rows.append(metrics)
        results[config_name] = sweep_rows

    return results


# ------------------------------------------------------------------
# Printing helpers
# ------------------------------------------------------------------

def print_sweep_table(
    sweep_results: Dict[str, List[Dict[str, float]]],
    config_name: str,
) -> None:
    """Print a single config's threshold sweep as a formatted table."""
    rows = sweep_results[config_name]
    print(f"\n{'='*90}")
    print(f"THRESHOLD SWEEP — {config_name}")
    print(f"{'='*90}")
    print(f"  {'|z|':>5s}  {'Sens':>7s}  {'Spec':>7s}  {'FAR':>7s}  "
          f"{'PPV':>7s}  {'Acc':>7s}  {'Flagged':>8s}  "
          f"{'Chang→Flag':>11s}  {'Same→OK':>8s}")
    print(f"  {'-'*82}")
    for r in rows:
        print(f"  {r['threshold']:>5.2f}  "
              f"{r['sensitivity']:>7.3f}  "
              f"{r['specificity']:>7.3f}  "
              f"{r['FAR']:>7.3f}  "
              f"{r['PPV']:>7.3f}  "
              f"{r['accuracy']:>7.3f}  "
              f"{r['flagged']:>8d}  "
              f"{r['n_changed_flagged']:>5d}/{r['n_changed']:<5d}  "
              f"{r['n_same_not_flagged']:>5d}/{r['n_same']:<5d}")
    print(f"{'='*90}")


def print_comparison_table(
    sweep_4f: List[Dict[str, float]],
    sweep_3f: List[Dict[str, float]],
) -> None:
    """Print side-by-side 4-feature vs 3-feature comparison."""
    print(f"\n{'='*110}")
    print(f"SIDE-BY-SIDE COMPARISON: 4-feature (with QRS) vs 3-feature (without QRS)")
    print(f"{'='*110}")
    print(f"  {'':>5s}  {'─── 4-feature ───':^30s}  │  {'─── 3-feature ───':^30s}  │  {'Δ Sens':>8s}  {'Δ Spec':>8s}  {'Δ PPV':>8s}")
    print(f"  {'|z|':>5s}  {'Sens':>7s}  {'Spec':>7s}  {'PPV':>7s}  │  "
          f"{'Sens':>7s}  {'Spec':>7s}  {'PPV':>7s}  │  {'':>8s}  {'':>8s}  {'':>8s}")
    print(f"  {'-'*104}")

    for r4, r3 in zip(sweep_4f, sweep_3f):
        assert r4["threshold"] == r3["threshold"]
        ds = r3["sensitivity"] - r4["sensitivity"]
        dsp = r3["specificity"] - r4["specificity"]
        dp = r3["PPV"] - r4["PPV"]
        print(f"  {r4['threshold']:>5.2f}  "
              f"{r4['sensitivity']:>7.3f}  {r4['specificity']:>7.3f}  {r4['PPV']:>7.3f}  │  "
              f"{r3['sensitivity']:>7.3f}  {r3['specificity']:>7.3f}  {r3['PPV']:>7.3f}  │  "
              f"{ds:>+8.3f}  {dsp:>+8.3f}  {dp:>+8.3f}")

    print(f"{'='*110}")
    print(f"  Δ = 3-feature MINUS 4-feature (positive = adding QRS width hurts)")
    print()


def print_recommendation(
    sweep_4f: List[Dict[str, float]],
    sweep_3f: List[Dict[str, float]],
) -> None:
    """State a plain recommendation."""
    # Compute average absolute deltas across thresholds
    sens_deltas = [r3["sensitivity"] - r4["sensitivity"]
                   for r4, r3 in zip(sweep_4f, sweep_3f)]
    spec_deltas = [r3["specificity"] - r4["specificity"]
                   for r4, r3 in zip(sweep_4f, sweep_3f)]
    ppv_deltas = [r3["PPV"] - r4["PPV"]
                  for r4, r3 in zip(sweep_4f, sweep_3f)]

    mean_sens_delta = np.mean(sens_deltas)
    mean_spec_delta = np.mean(spec_deltas)
    mean_ppv_delta = np.mean(ppv_deltas)
    max_sens_delta = max(abs(d) for d in sens_deltas)
    max_spec_delta = max(abs(d) for d in spec_deltas)

    print(f"\n{'='*90}")
    print(f"RECOMMENDATION")
    print(f"{'='*90}")
    print()
    print(f"  Mean sensitivity delta (3f - 4f): {mean_sens_delta:>+.4f}")
    print(f"  Mean specificity delta (3f - 4f): {mean_spec_delta:>+.4f}")
    print(f"  Mean PPV delta (3f - 4f):          {mean_ppv_delta:>+.4f}")
    print(f"  Max |sensitivity delta|:            {max_sens_delta:.4f}")
    print(f"  Max |specificity delta|:            {max_spec_delta:.4f}")
    print()

    # Interpret
    if max_sens_delta < 0.01 and max_spec_delta < 0.01:
        verdict = (
            "NEGLIGIBLE DIFFERENCE either way.  Removing QRS width barely\n"
            "  changes sensitivity or specificity at any threshold.  The\n"
            "  unvalidated QRS feature is not pulling its weight, but also\n"
            "  not actively hurting.\n\n"
            "  RECOMMENDATION: Drop QRS width from the drift feature set.\n"
            " 理由: It adds complexity and a known caveat (unvalidated,\n"
            "  below-literature values) with no measurable benefit.  The\n"
            "  3-feature set (RR + HR + R-peak amplitude) is simpler and\n"
            "  equally effective.  State in the deck: '4 features were\n"
            "  tested; QRS width was dropped after ablation showed no\n"
            "  contribution to drift discrimination.'"
        )
    elif mean_sens_delta > 0.02:
        verdict = (
            "DROPPING QRS WIDTH HURTS SENSITIVITY.  The unvalidated feature\n"
            "  is doing real work — removing it loses meaningful detection\n"
            "  of superclass changes.  Keep QRS width with an explicit\n"
            "  caveat: 'QRS width is an unvalidated proxy; values are\n"
            "  below clinical norms due to preprocessing attenuation, but\n"
            "  the feature contributes to drift discrimination.'"
        )
    elif mean_spec_delta < -0.02:
        verdict = (
            "DROPPING QRS WIDTH IMPROVES SPECIFICITY.  QRS width was\n"
            "  generating false alarms without adding true positives.\n"
            "  RECOMMENDATION: Drop QRS width.  The 3-feature set is\n"
            "  cleaner."
        )
    else:
        verdict = (
            "MIXED / INCONCLUSIVE.  The deltas are small but non-zero.\n"
            "  RECOMMENDATION: Drop QRS width for simplicity — the marginal\n"
            "  differences don't justify keeping an unvalidated feature."
        )

    print(f"  {verdict}")
    print(f"{'='*90}")


# ------------------------------------------------------------------
# JSON output
# ------------------------------------------------------------------

def save_sweep_json(
    sweep_3f: List[Dict[str, float]],
    patients: List[PatientDriftCohort],
    output_path: Path,
) -> None:
    """Save 3-feature sweep results to JSON file."""
    # Build sweep entries
    sweep = []
    for r in sweep_3f:
        sweep.append({
            "z_threshold": r["threshold"],
            "sensitivity": round(r["sensitivity"], 3),
            "specificity": round(r["specificity"], 3),
            "false_alarm_rate": round(r["FAR"], 3),
            "ppv": round(r["PPV"], 3),
            "accuracy": round(r["accuracy"], 3),
            "n_flagged": int(r["flagged"]),
            "changed_flagged": int(r["n_changed_flagged"]),
            "same_flagged": int(r["n_same"] - r["n_same_not_flagged"]),
        })

    output = {
        "description": "Phase 5d — 3-feature drift threshold sweep (post-ablation, QRS width removed). 2,109 patients, 2,926 follow-ups. Features: RR interval, heart rate, R-peak amplitude (2 independent signals).",
        "n_patients": len(patients),
        "n_followups": sum(len(p.follow_ups) for p in patients),
        "n_superclass_changed": sum(1 for p in patients for fu in p.follow_ups
                                     if set(fu.superclasses) != set(p.baseline.superclasses)),
        "n_superclass_same": sum(1 for p in patients for fu in p.follow_ups
                                  if set(fu.superclasses) == set(p.baseline.superclasses)),
        "base_rate_changed": sum(1 for p in patients for fu in p.follow_ups
                                  if set(fu.superclasses) != set(p.baseline.superclasses)) / sum(len(p.follow_ups) for p in patients),
        "features": ["mean_rr_ms", "heart_rate_bpm", "mean_r_amplitude_mv"],
        "sweep": sweep,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved 3-feature sweep JSON to {output_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5d — Ablation: drift detection without QRS width")
    parser.add_argument("--output-json", type=Path, help="Path to save 3-feature sweep JSON")
    args = parser.parse_args()

    print("=" * 90)
    print("PHASE 5d — ABLATION: DRIFT DETECTION WITHOUT QRS WIDTH")
    print("=" * 90)
    print()

    t0 = time.time()

    # Step 1: Build cohort
    print("--- Step 1: Building drift cohort ---")
    patients, meta = build_drift_cohort()
    print_cohort_summary(patients)

    # Step 2: Extract features ONCE
    print("\n--- Step 2: Extracting features (once, for all configs) ---")
    features_cache = _extract_all_features(patients, _WFDB_DIR)

    # Step 3: Define feature configs
    FEAT_4 = ("mean_rr_ms", "heart_rate_bpm", "qrs_width_ms", "mean_r_amplitude_mv")
    FEAT_3 = ("mean_rr_ms", "heart_rate_bpm", "mean_r_amplitude_mv")

    # Redundancy note
    print("\n--- Redundancy note ---")
    print("  RR interval and HR are mathematically linked: HR = 60000 / RR_ms.")
    print("  They are NOT independent signals — they carry the same information")
    print("  scaled differently.  The 3-feature version has effectively")
    print("  2 INDEPENDENT signals: (1) cardiac rate (RR/HR) and (2) R-peak")
    print("  amplitude.  Both the 4-feature and 3-feature versions have this")
    print("  redundancy; dropping QRS width does NOT change it.")
    print("  The ablation tests whether QRS width adds value BEYOND rate + amplitude.")

    # Step 4: Threshold sweep
    THRESHOLDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    feature_configs = {
        "4-feature (RR + HR + QRS + Amp)": FEAT_4,
        "3-feature (RR + HR + Amp)": FEAT_3,
    }

    print("\n--- Step 3: Threshold sweep ---")
    sweep_results = threshold_sweep(patients, features_cache, feature_configs, THRESHOLDS)

    # Step 5: Print tables
    for config_name in feature_configs:
        print_sweep_table(sweep_results, config_name)

    # Step 6: Side-by-side comparison
    print("\n--- Step 4: Side-by-side comparison ---")
    sweep_4f = sweep_results["4-feature (RR + HR + QRS + Amp)"]
    sweep_3f = sweep_results["3-feature (RR + HR + Amp)"]
    print_comparison_table(sweep_4f, sweep_3f)

    # Step 7: Recommendation
    print_recommendation(sweep_4f, sweep_3f)

    # Step 8: Save JSON if requested
    if args.output_json:
        save_sweep_json(sweep_3f, patients, args.output_json)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
