"""Cadence ECG Demo Dashboard (Phase 7 + Phase 8).

Streamlit app that wires together:
  - Trained + calibrated classifier (Phase 3/4)
  - Reject/abstention policy (Phase 4)
  - Drift layer PoC (Phase 5, clearly labeled)
  - Latency benchmark results (Phase 6)
  - Interactive risk-threshold sliders (Phase 8)

Run:  streamlit run app.py

SCOPE (PRD §12): This is a demo dashboard, not a clinical tool.
All outputs are research-stage prototype quality per §9 and §11.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is on sys.path
REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.dataset import ECGDataset, TARGET_LENGTH
from src.data.loader import SUPERCLASSES
from src.data.wfdb import read_wfdb_record
from src.models.baseline_classifier import ECGClassifier1D
from src.preprocessing.filters import preprocess

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
CHECKPOINT = REPO / "checkpoints" / "best_model.pt"
TEMPERATURE_PATH = REPO / "checkpoints" / "calibration_temperature.json"
POLICY_PATH = REPO / "checkpoints" / "reject_policy.json"
LATENCY_JSON = REPO / "notebooks" / "latency_qc" / "latency_benchmark.json"
DRIFT_SWEEP_JSON = REPO / "notebooks" / "latency_qc" / "drift_sweep_3feature.json"
WFDB_DIR = REPO / "data" / "ptbxl" / "wfdb"

FS = 500.0

# ------------------------------------------------------------------
# PRD §9 disclaimer — always visible, never softened
# ------------------------------------------------------------------
DISCLAIMER = """
> **⚠️ Research Prototype — Not a Clinical Tool**
> This system **triages and flags recordings for human review**; it does not
> diagnose.  It is **not clinically validated**, not cleared by any regulatory
> body, and should not be used to make treatment decisions.  All outputs are
> from a research-stage proof of concept (PRD §9, §11).
"""

# ------------------------------------------------------------------
# Cached model loading
# ------------------------------------------------------------------

@st.cache_resource
def load_model():
    """Load the trained model, temperature, and reject policy."""
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = ECGClassifier1D(n_leads=12, n_classes=len(SUPERCLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    temperature = 1.0
    if TEMPERATURE_PATH.exists():
        temperature = json.loads(TEMPERATURE_PATH.read_text())["temperature"]

    class_thresholds = {}
    if POLICY_PATH.exists():
        policy = json.loads(POLICY_PATH.read_text())
        class_thresholds = policy.get("class_thresholds", {})

    return model, temperature, class_thresholds


@st.cache_resource
def load_test_dataset():
    """Load the test-split dataset (cached preprocessed waveforms)."""
    return ECGDataset("test")


@st.cache_data
def build_drift_index():
    """Build a patient_id → baseline_ecg_id lookup from the drift cohort."""
    from src.data.drift_cohort import build_drift_cohort
    patients, _ = build_drift_cohort()
    index = {}
    for p in patients:
        index[p.patient_id] = {
            "baseline_ecg_id": p.baseline.ecg_id,
            "baseline_superclasses": p.baseline.superclasses,
            "baseline_date": p.baseline.recording_date,
            "follow_up_ecg_ids": [fu.ecg_id for fu in p.follow_ups],
        }
    return index


@st.cache_data
def load_latency_results():
    """Load precomputed latency benchmark results."""
    if LATENCY_JSON.exists():
        return json.loads(LATENCY_JSON.read_text())
    return None


@st.cache_data
def load_test_db():
    """Load the test-split database metadata."""
    ds = ECGDataset("test")
    return ds.db.copy()


@st.cache_data
def load_reject_curve():
    """Load Phase 4 precomputed reject-policy threshold-coverage curve."""
    if not POLICY_PATH.exists():
        return None
    policy = json.loads(POLICY_PATH.read_text())
    return policy.get("curve", [])


@st.cache_data
def load_drift_sweep():
    """Load Phase 5d precomputed 3-feature drift threshold sweep."""
    if not DRIFT_SWEEP_JSON.exists():
        return None
    return json.loads(DRIFT_SWEEP_JSON.read_text())


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

def run_inference(model, signal_tensor, temperature, class_thresholds):
    """Run end-to-end inference on a preprocessed signal tensor."""
    x = signal_tensor.unsqueeze(0)  # (1, 12, T)
    with torch.no_grad():
        logits = model(x)
    logits_np = logits.squeeze(0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_np / temperature))
    predicted = (probs >= 0.5).astype(int)
    confidence = np.maximum(probs, 1.0 - probs)
    thresholds = np.array([class_thresholds.get(name, 0.75) for name in SUPERCLASSES])
    keep = confidence >= thresholds
    return {
        "probabilities": probs,
        "predicted": predicted,
        "confidence": confidence,
        "keep": keep,
    }


# ------------------------------------------------------------------
# Waveform display
# ------------------------------------------------------------------

def plot_12_lead_waveform(signal_mV, fs=FS, title="12-Lead ECG"):
    """Plot a 3×4 grid of all 12 leads using matplotlib → st.pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    n_leads, n_samples = signal_mV.shape
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(4, 3, figsize=(12, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < n_leads:
            ax.plot(t, signal_mV[i], linewidth=0.6, color="#2c3e50")
            ax.set_ylabel(lead_names[i], fontsize=9, fontweight="bold")
            ax.set_ylim(-2.0, 2.0)
            ax.axhline(0, color="gray", linewidth=0.3, linestyle="--")
        ax.set_xticks(np.arange(0, t[-1] + 0.1, 1.0))
        ax.tick_params(labelsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
# Phase 8: Interactive tradeoff charts
# ------------------------------------------------------------------

def render_tradeoff_tab():
    """Render the interactive risk-threshold slider tab (Phase 8)."""
    st.header("🎚️ Try the Tradeoff")
    st.markdown("""
    Drag the slider to explore the clinical tradeoff between **catching more
    abnormalities** and **reducing false alarms**. Both charts use **precomputed,
    validated table values** — no metrics are recomputed at runtime.
    """)

    # ── Reject Policy Tradeoff (Phase 4) ──
    st.subheader("1. Confidence-Reject Tradeoff (Population Classifier)")
    st.caption("Source: Phase 4 precomputed threshold-coverage curve — "
               "higher threshold = fewer recordings auto-classified, "
               "but higher accuracy on those kept.")

    reject_curve = load_reject_curve()
    if reject_curve:
        thresholds = [row["threshold"] for row in reject_curve]
        coverages = [row["coverage"] * 100 for row in reject_curve]
        accuracies = [row["accuracy_kept"] * 100 for row in reject_curve]
        errors = [row["error_kept"] * 100 for row in reject_curve]

        # Slider snaps to real threshold values only
        slider_idx = st.slider(
            "Confidence threshold",
            min_value=0,
            max_value=len(thresholds) - 1,
            value=4,  # default to 0.7 (index 4)
            step=1,
            format_func=lambda i: f"{thresholds[i]:.2f}",
            key="reject_slider",
            help="Snaps to real precomputed threshold values only — no interpolation."
        )

        sel = reject_curve[slider_idx]
        cov_pct = sel["coverage"] * 100
        acc_pct = sel["accuracy_kept"] * 100
        err_pct = sel["error_kept"] * 100
        review_pct = (1.0 - sel["coverage"]) * 100

        # Chart: coverage and accuracy vs threshold, with current point highlighted
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Coverage + Accuracy
        ax1.plot(thresholds, coverages, "o-", color="#3498db",
                 linewidth=2, markersize=6, label="Coverage (%)")
        ax1.plot(thresholds, accuracies, "s-", color="#2ecc71",
                 linewidth=2, markersize=6, label="Accuracy on kept (%)")
        ax1.axvline(thresholds[slider_idx], color="#e74c3c",
                    linewidth=1.5, linestyle="--", alpha=0.7,
                    label=f"Selected: {thresholds[slider_idx]:.2f}")
        ax1.plot(thresholds[slider_idx], coverages[slider_idx], "o",
                 color="#e74c3c", markersize=12, zorder=5)
        ax1.plot(thresholds[slider_idx], accuracies[slider_idx], "s",
                 color="#e74c3c", markersize=12, zorder=5)
        ax1.set_xlabel("Confidence Threshold", fontsize=10)
        ax1.set_ylabel("Percentage (%)", fontsize=10)
        ax1.set_title("Coverage vs. Accuracy", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, loc="center left")
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)

        # Right: Error rate (what reject policy eliminates)
        ax2.plot(thresholds, errors, "D-", color="#e67e22",
                 linewidth=2, markersize=6, label="Error on kept (%)")
        ax2.axvline(thresholds[slider_idx], color="#e74c3c",
                    linewidth=1.5, linestyle="--", alpha=0.7)
        ax2.plot(thresholds[slider_idx], errors[slider_idx], "D",
                 color="#e74c3c", markersize=12, zorder=5)
        ax2.set_xlabel("Confidence Threshold", fontsize=10)
        ax2.set_ylabel("Error Rate (%)", fontsize=10)
        ax2.set_title("Error Rate on Kept Decisions", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 20)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Plain-language summary — numbers pulled directly from the table row
        st.markdown(f"""
        **At threshold {sel['threshold']:.2f}:**
        - **{cov_pct:.1f}%** of class decisions are auto-classified (kept)
        - **{review_pct:.1f}%** are flagged for human review (rejected)
        - Accuracy on kept decisions: **{acc_pct:.1f}%**
        - Error rate on kept decisions: **{err_pct:.1f}%**
        """)
    else:
        st.error("Reject policy curve not found.")

    st.divider()

    # ── Drift Threshold Tradeoff (Phase 5d) ──
    st.subheader("2. Drift Detection Tradeoff (Phase 5d — PoC)")
    st.caption("⚠️ This is a proof-of-concept drift layer, NOT a validated feature (PRD §11). "
               "Source: Phase 5d precomputed 3-feature threshold sweep — "
               "|z| threshold = how sensitive the drift detector is.")

    drift_data = load_drift_sweep()
    if drift_data and "sweep" in drift_data:
        sweep = drift_data["sweep"]
        z_thresholds = [row["z_threshold"] for row in sweep]
        sensitivities = [row["sensitivity"] * 100 for row in sweep]
        specificities = [row["specificity"] * 100 for row in sweep]
        ppvs = [row["ppv"] * 100 for row in sweep]
        far_rates = [row["false_alarm_rate"] * 100 for row in sweep]

        # Slider snaps to real z-threshold values only
        drift_idx = st.slider(
            "Drift z-score threshold",
            min_value=0,
            max_value=len(z_thresholds) - 1,
            value=3,  # default to 1.0 (index 3)
            step=1,
            format_func=lambda i: f"|z| > {z_thresholds[i]:.2f}",
            key="drift_slider",
            help="Snaps to real precomputed z-threshold values only — no interpolation."
        )

        sel = sweep[drift_idx]
        sens = sel["sensitivity"] * 100
        spec = sel["specificity"] * 100
        ppv = sel["ppv"] * 100
        far = sel["false_alarm_rate"] * 100
        flagged = sel["n_flagged"]
        changed = drift_data["n_superclass_changed"]
        same = drift_data["n_superclass_same"]

        # Chart: sensitivity/specificity/PPV vs z-threshold
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Left: Sensitivity + Specificity
        ax1.plot(z_thresholds, sensitivities, "o-", color="#e74c3c",
                 linewidth=2, markersize=6, label="Sensitivity (%)")
        ax1.plot(z_thresholds, specificities, "s-", color="#3498db",
                 linewidth=2, markersize=6, label="Specificity (%)")
        ax1.axvline(z_thresholds[drift_idx], color="#9b59b6",
                    linewidth=1.5, linestyle="--", alpha=0.7,
                    label=f"Selected: |z|>{z_thresholds[drift_idx]:.2f}")
        ax1.plot(z_thresholds[drift_idx], sensitivities[drift_idx], "o",
                 color="#e74c3c", markersize=12, zorder=5)
        ax1.plot(z_thresholds[drift_idx], specificities[drift_idx], "s",
                 color="#3498db", markersize=12, zorder=5)
        ax1.set_xlabel("|z| Threshold", fontsize=10)
        ax1.set_ylabel("Percentage (%)", fontsize=10)
        ax1.set_title("Sensitivity vs. Specificity", fontsize=11, fontweight="bold")
        ax1.legend(fontsize=8, loc="center right")
        ax1.set_ylim(0, 105)
        ax1.grid(True, alpha=0.3)

        # Right: PPV + False Alarm Rate
        ax2.plot(z_thresholds, ppvs, "^-", color="#2ecc71",
                 linewidth=2, markersize=6, label="PPV (%)")
        ax2.plot(z_thresholds, far_rates, "v-", color="#e67e22",
                 linewidth=2, markersize=6, label="False Alarm Rate (%)")
        ax2.axvline(z_thresholds[drift_idx], color="#9b59b6",
                    linewidth=1.5, linestyle="--", alpha=0.7)
        ax2.plot(z_thresholds[drift_idx], ppvs[drift_idx], "^",
                 color="#2ecc71", markersize=12, zorder=5)
        ax2.plot(z_thresholds[drift_idx], far_rates[drift_idx], "v",
                 color="#e67e22", markersize=12, zorder=5)
        ax2.set_xlabel("|z| Threshold", fontsize=10)
        ax2.set_ylabel("Percentage (%)", fontsize=10)
        ax2.set_title("PPV vs. False Alarm Rate", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8, loc="center right")
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Plain-language summary — numbers pulled directly from the table row
        st.markdown(f"""
        **At |z| > {sel['z_threshold']:.2f}** (3-feature: RR, HR, R-peak amplitude):
        - **{sens:.1f}%** of real superclass changes are caught (sensitivity)
        - **{spec:.1f}%** of same-superclass cases correctly left alone (specificity)
        - **{ppv:.1f}%** of flagged cases truly changed (PPV)
        - **{far:.1f}%** of same-superclass cases falsely flagged (false alarm rate)
        - **{flagged}** recordings flagged out of {drift_data['n_followups']} follow-ups
          ({changed} actually changed, {same} stayed the same)

        *⚠️ Base rate of "superclass changed" is {drift_data['base_rate_changed']*100:.1f}% — a trivial
        always-predict-same classifier gets ~52% accuracy for free. The drift layer
        sits close to that floor. This is a benchmarked PoC finding, not a clinically
        useful triage feature (PRD §11).*
        """)
    else:
        st.error("Drift sweep data not found.")


# ------------------------------------------------------------------
# Static fallback content
# ------------------------------------------------------------------

def render_static_fallback():
    """Render the static results view (PRD §11 risk mitigation)."""
    st.header("📊 Static Results (No Live Inference Required)")

    st.markdown("""
    This tab shows precomputed results from Phases 4-6.
    It works independently of live inference — use this if the live
    inference tab encounters issues during demonstration.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Phase 4: Test-Set Performance")
        st.markdown("""
        | Metric | Value |
        |---|---|
        | Test-set accuracy (micro-avg) | 85.6% |
        | After reject (threshold 0.75) | 92.1% |
        | Coverage (decisions kept) | 76.3% |
        """)

    with col2:
        st.subheader("Phase 6: Latency (CPU)")
        latency = load_latency_results()
        if latency:
            sr = latency["single_recording"]
            st.markdown(f"""
            | Metric | Value |
            |---|---|
            | End-to-end mean | {sr['total']['mean_ms']:.2f} ms |
            | End-to-end p99 | {sr['total']['p99_ms']:.2f} ms |
            | Preprocessing | {sr['preprocessing']['mean_ms']:.2f} ms |
            | Model inference | {sr['inference']['mean_ms']:.2f} ms |
            | Hardware | {latency['hardware']['cpu_model']} |
            """)
        else:
            st.info("Latency results not found.")

    st.subheader("Phase 5: Drift Layer (PoC)")
    st.markdown("""
    | Metric | Value |
    |---|---|
    | Cohort | 2,109 patients, 2,926 follow-ups |
    | Sensitivity (|z|>1.0) | 15.3% |
    | Specificity (|z|>1.0) | 89.3% |
    | PPV | 57.3% |
    | Assessment | Real but weak signal — PoC, not validated |
    """)

    st.subheader("Example Waveforms")
    ds = load_test_dataset()
    db = ds.db
    st.caption("Showing 4 example test-set recordings (preprocessed waveforms).")

    examples = []
    for sc in SUPERCLASSES:
        matches = db[db["superclass"].apply(lambda x: sc in x)]
        if len(matches) > 0:
            examples.append(matches.iloc[0])
    while len(examples) < 4 and len(db) > len(examples):
        rng = np.random.default_rng(42)
        idx = rng.integers(len(db))
        if db.iloc[idx]["ecg_id"] not in [e["ecg_id"] for e in examples]:
            examples.append(db.iloc[idx])

    for i, row in enumerate(examples[:4]):
        idx_in_ds = db[db["ecg_id"] == row["ecg_id"]].index[0]
        signal, _, _ = ds[idx_in_ds]
        signal_np = signal.numpy() if isinstance(signal, torch.Tensor) else signal
        sc_label = row["superclass"]
        if isinstance(sc_label, list):
            sc_label = ", ".join(sc_label) if sc_label else "none"
        fig = plot_12_lead_waveform(signal_np, title=f"ecg_id={row['ecg_id']} — {sc_label}")
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Cadence ECG Dashboard",
        page_icon="🫀",
        layout="wide",
    )

    # ── Disclaimer banner (always visible, PRD §9) ──
    st.markdown(DISCLAIMER)
    st.divider()

    # ── Tabs ──
    tab_live, tab_static, tab_tradeoff = st.tabs(
        ["🔬 Live Inference", "📊 Static Results", "🎚️ Try the Tradeoff"]
    )

    # ════════════════════════════════════════════════════════════════
    # TAB 1: Live Inference
    # ════════════════════════════════════════════════════════════════
    with tab_live:
        st.header("Live Inference on Test-Set Recording")

        # Load resources
        model, temperature, class_thresholds = load_model()
        dataset = load_test_dataset()
        drift_index = build_drift_index()
        db = dataset.db

        st.caption(f"Test split: {len(db)} recordings. "
                   "Recording selector draws from TEST split only (never train/val).")

        # ── Recording selector ──
        ecg_ids = db["ecg_id"].tolist()
        patient_ids = db["patient_id"].tolist()
        superclasses_col = db["superclass"].tolist()

        def format_rec(idx):
            eid = ecg_ids[idx]
            pid = patient_ids[idx]
            sc = superclasses_col[idx]
            if isinstance(sc, list):
                sc_str = ", ".join(sc) if sc else "none"
            else:
                sc_str = str(sc)
            return f"ecg_id={eid}  |  patient={int(pid)}  |  {sc_str}"

        selected_display = st.selectbox(
            "Select a test-set recording:",
            options=list(range(len(db))),
            format_func=format_rec,
            key="recording_selector",
        )

        selected_row = db.iloc[selected_display]
        selected_ecg_id = int(selected_row["ecg_id"])
        selected_patient_id = int(selected_row["patient_id"])
        selected_sc = selected_row["superclass"]
        if isinstance(selected_sc, list):
            selected_sc_str = ", ".join(selected_sc) if selected_sc else "none"
        else:
            selected_sc_str = str(selected_sc)

        st.markdown(f"**Selected:** ecg_id={selected_ecg_id} | "
                    f"patient_id={selected_patient_id} | "
                    f"ground truth: {selected_sc_str}")

        # ── Load and display waveform ──
        st.subheader("Waveform (preprocessed — Phase 2 bandpass + wavelet)")
        signal_tensor, label_tensor, ecg_id_out = dataset[selected_display]
        signal_np = signal_tensor.numpy()

        fig = plot_12_lead_waveform(signal_np, title=f"ecg_id={selected_ecg_id}")
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)

        # ── Run inference ──
        st.subheader("Classification Result")
        result = run_inference(model, signal_tensor, temperature, class_thresholds)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Calibrated Probabilities & Predictions:**")
            for i, name in enumerate(SUPERCLASSES):
                prob = result["probabilities"][i]
                pred = "✅ YES" if result["predicted"][i] else "❌ NO"
                conf = result["confidence"][i]
                kept = "✅ kept" if result["keep"][i] else "⚠️ **needs review**"
                st.markdown(
                    f"- **{name}**: {prob:.3f} (pred={pred}, "
                    f"conf={conf:.3f}, {kept})"
                )

        with col_b:
            st.markdown("**Summary:**")
            n_predicted = int(result["predicted"].sum())
            n_keep = int(result["keep"].sum())
            st.metric("Classes predicted positive", n_predicted)
            st.metric("Decisions kept (confidence ≥ threshold)", f"{n_keep}/5")
            if n_keep < 5:
                st.warning(
                    f"⚠️ {5 - n_keep} class decision(s) flagged as 'needs review' "
                    f"(confidence below reject threshold). This recording should "
                    f"be reviewed by a clinician."
                )
            else:
                st.success("All 5 class decisions passed the confidence threshold.")

        # ── Drift section ──
        st.subheader("Drift Analysis (Phase 5 — PoC)")
        st.caption("⚠️ This is a proof-of-concept drift layer, NOT a validated feature. "
                   "See PRD §11.")

        if selected_patient_id in drift_index:
            info = drift_index[selected_patient_id]
            baseline_ecg = info["baseline_ecg_id"]
            baseline_sc = info["baseline_superclasses"]
            baseline_sc_str = ", ".join(baseline_sc) if baseline_sc else "none"
            n_followups = len(info["follow_up_ecg_ids"])

            st.markdown(
                f"**Patient {selected_patient_id} has a baseline recording** "
                f"(ecg_id={baseline_ecg}, date={info['baseline_date']}, "
                f"superclasses={baseline_sc_str})."
            )
            st.markdown(f"Patient has {n_followups} follow-up recording(s) in the drift cohort.")

            if selected_ecg_id == baseline_ecg:
                st.info("This recording IS the patient's baseline (first recording).")
            elif selected_ecg_id in info["follow_up_ecg_ids"]:
                from src.data.wfdb import read_wfdb_record as _read_wfdb_record
                from src.preprocessing.beat_features import extract_beat_features

                baseline_fname = None
                fu_fname = None
                for _, row in dataset.db.iterrows():
                    if int(row["ecg_id"]) == baseline_ecg:
                        baseline_fname = row["filename_hr"]
                    if int(row["ecg_id"]) == selected_ecg_id:
                        fu_fname = row["filename_hr"]

                if baseline_fname and fu_fname:
                    try:
                        b_hea = WFDB_DIR / (baseline_fname + ".hea")
                        b_dat = WFDB_DIR / (baseline_fname + ".dat")
                        b_sig, b_fs, _ = _read_wfdb_record(b_hea, b_dat)
                        b_proc = preprocess(b_sig, b_fs)
                        b_feat = extract_beat_features(b_proc, fs=b_fs)

                        f_hea = WFDB_DIR / (fu_fname + ".hea")
                        f_dat = WFDB_DIR / (fu_fname + ".dat")
                        f_sig, f_fs, _ = _read_wfdb_record(f_hea, f_dat)
                        f_proc = preprocess(f_sig, f_fs)
                        f_feat = extract_beat_features(f_proc, fs=f_fs)

                        from src.models.drift import DriftConfig, compute_z_scores
                        cfg = DriftConfig(z_threshold=1.0, features_to_check=(
                            "mean_rr_ms", "heart_rate_bpm", "mean_r_amplitude_mv"))
                        z_scores = compute_z_scores(b_feat, f_feat, cfg)
                        max_abs_z = max(abs(z) for z in z_scores.values())
                        flagged = max_abs_z > cfg.z_threshold

                        if flagged:
                            st.error(
                                f"🔴 **Drift FLAGGED** (|z|={max_abs_z:.3f} > {cfg.z_threshold})\n\n"
                                f"Feature z-scores: " +
                                ", ".join(f"{k}={v:.3f}" for k, v in z_scores.items())
                            )
                        else:
                            st.success(
                                f"🟢 **No drift** (|z|={max_abs_z:.3f} ≤ {cfg.z_threshold})\n\n"
                                f"Feature z-scores: " +
                                ", ".join(f"{k}={v:.3f}" for k, v in z_scores.items())
                            )
                    except Exception as e:
                        st.warning(f"Could not compute drift: {e}")
                else:
                    st.info("Could not locate waveform files for drift comparison.")
            else:
                st.info("This recording is neither the baseline nor a follow-up in the drift cohort.")
        else:
            st.info(
                f"**No prior baseline available for patient {selected_patient_id}.** "
                f"This patient has only one recording in the dataset, so drift "
                f"detection cannot be performed."
            )

    # ════════════════════════════════════════════════════════════════
    # TAB 2: Static Fallback
    # ════════════════════════════════════════════════════════════════
    with tab_static:
        render_static_fallback()

    # ════════════════════════════════════════════════════════════════
    # TAB 3: Interactive Tradeoff (Phase 8)
    # ════════════════════════════════════════════════════════════════
    with tab_tradeoff:
        render_tradeoff_tab()


if __name__ == "__main__":
    main()
