"""Phase 6 — Real-time per-beat inference latency benchmark (PRD §8).

Measures end-to-end inference latency of the population classifier under
conditions that approximate real-time streaming use at a PHC/CHC:

  1. Single-recording latency: raw waveform in → preprocessing → model
     forward pass → calibrated confidence + reject decision out.
  2. Streaming playback simulation: feed a recording in real-time-paced
     chunks, measure latency from "last sample of a beat arrives" to
     "classification available."

SCOPE LIMITATION (PRD §5 non-goals):
  This is software-only, simulated streaming via recorded signal playback
  — NOT real hardware/device integration.  The numbers measure what the
  software stack can achieve, not what a live device would deliver.

CPU vs GPU:
  CPU is the realistic PHC/CHC deployment target per PRD §2.  All
  measurements are on CPU.  If GPU is available, it is stated but NOT
  reported as the primary number.

HARDWARE REPORTED:
  Real spec of the machine that ran the benchmark, not assumed.

Usage:
  .venv/bin/python -m src.eval.latency_benchmark
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.data.dataset import ECGDataset, TARGET_LENGTH
from src.data.loader import SUPERCLASSES
from src.models.baseline_classifier import ECGClassifier1D
from src.preprocessing.filters import preprocess

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent.parent
CHECKPOINT = REPO / "checkpoints" / "best_model.pt"
TEMPERATURE_PATH = REPO / "checkpoints" / "calibration_temperature.json"
POLICY_PATH = REPO / "checkpoints" / "reject_policy.json"
RESULTS_DIR = REPO / "notebooks" / "latency_qc"

FS = 500.0  # PTB-XL high-resolution sampling rate


# ------------------------------------------------------------------
# Hardware info
# ------------------------------------------------------------------

@dataclass
class HardwareInfo:
    cpu_model: str
    n_physical_cores: int
    n_logical_cores: int
    ram_gb: float
    os: str
    python: str
    torch_version: str
    has_mps: bool  # Apple Silicon GPU
    has_cuda: bool


def get_hardware_info() -> HardwareInfo:
    """Detect actual hardware specs."""
    cpu_model = "unknown"
    try:
        if platform.system() == "Darwin":
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, timeout=5
            ).strip()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":")[1].strip()
                        break
    except Exception:
        pass

    n_logical = os.cpu_count() or 1
    n_physical = n_logical
    try:
        if platform.system() == "Darwin":
            n_physical = int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"],
                text=True, timeout=5
            ).strip())
    except Exception:
        pass

    ram_gb = 0.0
    try:
        if platform.system() == "Darwin":
            ram_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True, timeout=5
            ).strip())
            ram_gb = ram_bytes / (1024 ** 3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        ram_kb = int(line.split()[1])
                        ram_gb = ram_kb / (1024 ** 2)
                        break
    except Exception:
        pass

    return HardwareInfo(
        cpu_model=cpu_model,
        n_physical_cores=n_physical,
        n_logical_cores=n_logical,
        ram_gb=round(ram_gb, 1),
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        torch_version=torch.__version__,
        has_mps=torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
        has_cuda=torch.cuda.is_available(),
    )


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model_and_policy() -> Tuple[ECGClassifier1D, float, Dict[str, float]]:
    """Load the trained model, temperature, and reject policy.

    Returns (model, temperature, class_thresholds).
    Model is set to eval mode on CPU.
    """
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

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


# ------------------------------------------------------------------
# Preprocessing helper (raw waveform → model-ready)
# ------------------------------------------------------------------

def preprocess_raw(raw_signal: np.ndarray, fs: float = FS) -> np.ndarray:
    """Apply Phase 2 preprocessing (bandpass + wavelet) and pad/truncate.

    Input: (n_leads, n_samples) raw waveform in mV.
    Output: (12, TARGET_LENGTH) preprocessed float32 tensor-ready array.
    """
    processed = preprocess(raw_signal, fs)
    n_leads, n_samples = processed.shape
    if n_samples < TARGET_LENGTH:
        processed = np.pad(
            processed, ((0, 0), (0, TARGET_LENGTH - n_samples)),
            mode="constant",
        )
    elif n_samples > TARGET_LENGTH:
        processed = processed[:, :TARGET_LENGTH]
    return processed.astype(np.float32)


# ------------------------------------------------------------------
# End-to-end inference
# ------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Result of a single end-to-end inference pass."""
    probabilities: np.ndarray   # (5,) calibrated probabilities
    predicted: np.ndarray       # (5,) binary predictions at 0.5
    confidence: np.ndarray      # (5,) decision confidence
    keep: np.ndarray            # (5,) which classes pass reject threshold
    preprocess_ms: float        # Time for preprocessing
    inference_ms: float         # Time for model forward pass
    calibration_ms: float       # Time for sigmoid + temperature scaling
    reject_ms: float            # Time for reject decision
    total_ms: float             # End-to-end time


def run_inference(
    model: ECGClassifier1D,
    raw_signal: np.ndarray,
    temperature: float,
    class_thresholds: Dict[str, float],
    fs: float = FS,
) -> InferenceResult:
    """Run end-to-end inference: preprocess → forward → calibrate → reject.

    All timings are on CPU.
    """
    # 1. Preprocessing
    t0 = time.perf_counter()
    processed = preprocess_raw(raw_signal, fs)
    t1 = time.perf_counter()
    preprocess_ms = (t1 - t0) * 1000.0

    # 2. Model forward pass
    x = torch.from_numpy(processed).unsqueeze(0)  # (1, 12, T)
    t2 = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
    t3 = time.perf_counter()
    inference_ms = (t3 - t2) * 1000.0

    # 3. Calibration (sigmoid + temperature scaling)
    t4 = time.perf_counter()
    logits_np = logits.squeeze(0).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_np / temperature))
    t5 = time.perf_counter()
    calibration_ms = (t5 - t4) * 1000.0

    # 4. Reject decision
    t6 = time.perf_counter()
    predicted = (probs >= 0.5).astype(int)
    confidence = np.maximum(probs, 1.0 - probs)
    thresholds = np.array([class_thresholds.get(name, 0.75) for name in SUPERCLASSES])
    keep = confidence >= thresholds
    t7 = time.perf_counter()
    reject_ms = (t7 - t6) * 1000.0

    total_ms = (t7 - t0) * 1000.0

    return InferenceResult(
        probabilities=probs,
        predicted=predicted,
        confidence=confidence,
        keep=keep,
        preprocess_ms=preprocess_ms,
        inference_ms=inference_ms,
        calibration_ms=calibration_ms,
        reject_ms=reject_ms,
        total_ms=total_ms,
    )


# ------------------------------------------------------------------
# Single-recording latency benchmark
# ------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Aggregated latency statistics."""
    n_samples: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


def compute_stats(values: np.ndarray) -> LatencyStats:
    """Compute latency statistics from an array of ms values."""
    return LatencyStats(
        n_samples=len(values),
        mean_ms=float(np.mean(values)),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        min_ms=float(np.min(values)),
        max_ms=float(np.max(values)),
        std_ms=float(np.std(values)),
    )


def benchmark_single_recording(
    model: ECGClassifier1D,
    dataset: ECGDataset,
    temperature: float,
    class_thresholds: Dict[str, float],
    n_samples: int = 200,
    warmup: int = 10,
) -> Tuple[LatencyStats, LatencyStats, LatencyStats, LatencyStats, HardwareInfo]:
    """Benchmark single-recording inference latency.

    Measures END-TO-END: raw waveform → preprocess → model forward →
    calibrate → reject.  All timings on CPU.

    Returns (total_stats, preprocess_stats, inference_stats, other_stats, hw_info).
    """
    hw_info = get_hardware_info()

    from src.data.wfdb import read_wfdb_record
    wfdb_dir = dataset.wfdb_dir

    # Select a random subset of test recordings and load their raw waveforms
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_samples + warmup, len(dataset)),
                         replace=False)

    # Load raw waveforms for the selected indices
    raw_recordings = []  # list of (raw_signal, fs)
    for idx in indices:
        row = dataset.db.iloc[int(idx)]
        hea_path = wfdb_dir / (row["filename_hr"] + ".hea")
        dat_path = wfdb_dir / (row["filename_hr"] + ".dat")
        try:
            signal, fs, _ = read_wfdb_record(hea_path, dat_path)
            raw_recordings.append((signal, fs))
        except Exception:
            continue

    print(f"    Loaded {len(raw_recordings)} raw waveforms for benchmarking", flush=True)

    total_times = []
    preprocess_times = []
    inference_times = []
    other_times = []

    # Warmup passes (full pipeline)
    for i in range(min(warmup, len(raw_recordings))):
        raw_signal, fs = raw_recordings[i]
        processed = preprocess_raw(raw_signal, fs)
        x = torch.from_numpy(processed).unsqueeze(0)
        with torch.no_grad():
            _ = model(x)

    # Benchmark full pipeline: raw → preprocess → model → calibrate → reject
    for raw_signal, fs in raw_recordings[warmup:warmup + n_samples]:
        # 1. Preprocessing
        t0 = time.perf_counter()
        processed = preprocess_raw(raw_signal, fs)
        t1 = time.perf_counter()

        # 2. Model forward pass
        x = torch.from_numpy(processed).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        t2 = time.perf_counter()

        # 3. Calibration (sigmoid + temperature)
        logits_np = logits.squeeze(0).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits_np / temperature))
        t3 = time.perf_counter()

        # 4. Reject decision
        predicted = (probs >= 0.5).astype(int)
        confidence = np.maximum(probs, 1.0 - probs)
        thresholds = np.array([class_thresholds.get(name, 0.75) for name in SUPERCLASSES])
        keep = confidence >= thresholds
        t4 = time.perf_counter()

        prep_ms = (t1 - t0) * 1000.0
        inf_ms = (t2 - t1) * 1000.0
        cal_ms = (t3 - t2) * 1000.0
        rej_ms = (t4 - t3) * 1000.0
        total_ms = (t4 - t0) * 1000.0

        preprocess_times.append(prep_ms)
        inference_times.append(inf_ms)
        total_times.append(total_ms)
        other_times.append(cal_ms + rej_ms)

    return (
        compute_stats(np.array(total_times)),
        compute_stats(np.array(preprocess_times)),
        compute_stats(np.array(inference_times)),
        compute_stats(np.array(other_times)),
        hw_info,
    )


# ------------------------------------------------------------------
# Streaming playback simulation
# ------------------------------------------------------------------

@dataclass
class StreamingResult:
    """Latency for one streaming-simulated beat classification."""
    beat_index: int
    chunk_size: int
    latency_ms: float  # Time from "last sample arrives" to "classification available"


def simulate_streaming(
    model: ECGClassifier1D,
    dataset: ECGDataset,
    temperature: float,
    class_thresholds: Dict[str, float],
    n_recordings: int = 100,
    chunk_ms: float = 100.0,
) -> Tuple[LatencyStats, List[StreamingResult]]:
    """Simulate real-time-paced streaming playback.

    Feeds the recording in time-paced chunks matching the sample rate.
    For each chunk, measures the latency from "chunk arrives" to
    "classification available" (model forward + calibrate + reject).

    The model processes the FULL recording up to the current chunk
    (because the 1D-CNN needs the full temporal context).  This is the
    realistic scenario: as new data arrives, the model re-classifies
    with the updated signal.

    chunk_ms: size of each chunk in milliseconds (default 100ms).

    Returns (overall_latency_stats, per_chunk_results).
    """
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_recordings, len(dataset)),
                         replace=False)

    chunk_samples = int(chunk_ms * FS / 1000.0)
    all_latencies = []
    all_results = []

    for rec_i, idx in enumerate(indices):
        idx = int(idx)
        signal_tensor, _, _ = dataset[idx]
        n_total_samples = signal_tensor.shape[1]

        # Simulate feeding chunks
        n_chunks = n_total_samples // chunk_samples
        for chunk_i in range(1, n_chunks + 1):
            end_sample = chunk_i * chunk_samples
            partial_signal = signal_tensor[:, :end_sample].unsqueeze(0)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(partial_signal)
            logits_np = logits.squeeze(0).numpy()
            probs = 1.0 / (1.0 + np.exp(-logits_np / temperature))
            predicted = (probs >= 0.5).astype(int)
            confidence = np.maximum(probs, 1.0 - probs)
            thresholds = np.array([class_thresholds.get(name, 0.75) for name in SUPERCLASSES])
            keep = confidence >= thresholds
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            all_latencies.append(latency_ms)
            all_results.append(StreamingResult(
                beat_index=chunk_i,
                chunk_size=end_sample,
                latency_ms=latency_ms,
            ))

    return compute_stats(np.array(all_latencies)), all_results


# ------------------------------------------------------------------
# Pretty printing
# ------------------------------------------------------------------

def print_stats(label: str, stats: LatencyStats) -> None:
    """Print latency statistics in a formatted block."""
    print(f"  {label}:")
    print(f"    n={stats.n_samples}  mean={stats.mean_ms:.2f}ms  "
          f"median={stats.median_ms:.2f}ms  p95={stats.p95_ms:.2f}ms  "
          f"p99={stats.p99_ms:.2f}ms")
    print(f"    min={stats.min_ms:.2f}ms  max={stats.max_ms:.2f}ms  "
          f"std={stats.std_ms:.2f}ms")


def print_hw_info(hw: HardwareInfo) -> None:
    """Print hardware information."""
    print(f"  Hardware:")
    print(f"    CPU:    {hw.cpu_model} ({hw.n_physical_cores}P/{hw.n_logical_cores}L cores)")
    print(f"    RAM:    {hw.ram_gb} GB")
    print(f"    OS:     {hw.os}")
    print(f"    Python: {hw.python}")
    print(f"    PyTorch: {hw.torch_version}")
    print(f"    MPS:    {'available' if hw.has_mps else 'not available'}")
    print(f"    CUDA:   {'available' if hw.has_cuda else 'not available'}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("PHASE 6 — REAL-TIME PER-BEAT INFERENCE LATENCY BENCHMARK")
    print("=" * 80)
    print()
    print("SCOPE: Software-only, simulated streaming via recorded signal playback.")
    print("       NOT real hardware/device integration (PRD §5 non-goal).")
    print("       CPU-only measurements (realistic PHC/CHC deployment target).")
    print()

    # Hardware
    hw_info = get_hardware_info()
    print_hw_info(hw_info)
    print()

    # Load model + policy
    print("--- Loading model and policy ---")
    model, temperature, class_thresholds = load_model_and_policy()
    print(f"  Checkpoint: {CHECKPOINT}")
    print(f"  Temperature: {temperature:.6f}")
    print(f"  Reject thresholds: {class_thresholds}")
    print()

    # Load test dataset
    print("--- Loading test dataset ---")
    test_ds = ECGDataset("test")
    print(f"  Test recordings: {len(test_ds)}")
    print()

    # --- Single-recording benchmark ---
    print("--- Benchmark 1: Single-recording latency ---")
    print(f"  Measuring on {min(200, len(test_ds))} test recordings (10 warmup)...")
    total_stats, preprocess_stats, inference_stats, other_stats, _ = \
        benchmark_single_recording(model, test_ds, temperature, class_thresholds,
                                   n_samples=200, warmup=10)

    print()
    print_stats("Total (preprocess + inference + calibrate + reject)", total_stats)
    print_stats("Preprocessing only (bandpass + wavelet)", preprocess_stats)
    print_stats("Inference only (model forward pass)", inference_stats)
    print_stats("Calibration + reject (sigmoid + temp scale + decision)", other_stats)

    # --- Streaming benchmark ---
    print()
    print("--- Benchmark 2: Streaming playback simulation ---")
    print(f"  Chunk size: 100ms (50 samples at 500Hz)")
    print(f"  Measuring on {min(100, len(test_ds))} test recordings...")
    streaming_stats, streaming_results = simulate_streaming(
        model, test_ds, temperature, class_thresholds,
        n_recordings=100, chunk_ms=100.0,
    )

    print()
    print_stats("Per-chunk classification latency (streaming)", streaming_stats)

    # --- Summary ---
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Single-recording total latency:")
    print(f"    mean={total_stats.mean_ms:.2f}ms  median={total_stats.median_ms:.2f}ms  "
          f"p95={total_stats.p95_ms:.2f}ms  p99={total_stats.p99_ms:.2f}ms")
    print(f"  Preprocessing breakdown: {preprocess_stats.mean_ms:.2f}ms mean "
          f"({preprocess_stats.mean_ms / total_stats.mean_ms * 100:.0f}% of total)")
    print(f"  Model inference breakdown: {inference_stats.mean_ms:.2f}ms mean "
          f"({inference_stats.mean_ms / total_stats.mean_ms * 100:.0f}% of total)")
    print(f"  Streaming per-chunk latency:")
    print(f"    mean={streaming_stats.mean_ms:.2f}ms  median={streaming_stats.median_ms:.2f}ms  "
          f"p95={streaming_stats.p95_ms:.2f}ms  p99={streaming_stats.p99_ms:.2f}ms")
    print()

    # --- Sanity framing ---
    print("SANITY FRAMING (per PRD §5):")
    print("  These are SOFTWARE-ONLY measurements via recorded signal playback.")
    print("  They do NOT represent real hardware/device integration latency.")
    print("  Real deployment would add device I/O, transmission, and UI latency.")
    print("  These numbers measure what the inference stack can achieve in isolation.")
    print("=" * 80)

    # Save results to disk
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "hardware": {
            "cpu_model": hw_info.cpu_model,
            "n_physical_cores": hw_info.n_physical_cores,
            "n_logical_cores": hw_info.n_logical_cores,
            "ram_gb": hw_info.ram_gb,
            "os": hw_info.os,
            "python": hw_info.python,
            "torch_version": hw_info.torch_version,
            "has_mps": hw_info.has_mps,
            "has_cuda": hw_info.has_cuda,
        },
        "single_recording": {
            "total": {"mean_ms": total_stats.mean_ms, "median_ms": total_stats.median_ms,
                       "p95_ms": total_stats.p95_ms, "p99_ms": total_stats.p99_ms,
                       "min_ms": total_stats.min_ms, "max_ms": total_stats.max_ms,
                       "std_ms": total_stats.std_ms, "n": total_stats.n_samples},
            "preprocessing": {"mean_ms": preprocess_stats.mean_ms, "median_ms": preprocess_stats.median_ms,
                              "p95_ms": preprocess_stats.p95_ms, "p99_ms": preprocess_stats.p99_ms,
                              "n": preprocess_stats.n_samples} if preprocess_stats else None,
            "inference": {"mean_ms": inference_stats.mean_ms, "median_ms": inference_stats.median_ms,
                          "p95_ms": inference_stats.p95_ms, "p99_ms": inference_stats.p99_ms,
                          "min_ms": inference_stats.min_ms, "max_ms": inference_stats.max_ms,
                          "n": inference_stats.n_samples},
        },
        "streaming": {
            "chunk_ms": 100.0,
            "mean_ms": streaming_stats.mean_ms,
            "median_ms": streaming_stats.median_ms,
            "p95_ms": streaming_stats.p95_ms,
            "p99_ms": streaming_stats.p99_ms,
            "min_ms": streaming_stats.min_ms,
            "max_ms": streaming_stats.max_ms,
            "n": streaming_stats.n_samples,
        },
        "model": {
            "checkpoint": str(CHECKPOINT),
            "temperature": temperature,
            "class_thresholds": class_thresholds,
        },
    }
    results_path = RESULTS_DIR / "latency_benchmark.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
