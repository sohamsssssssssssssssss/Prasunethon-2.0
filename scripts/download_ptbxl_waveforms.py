#!/usr/bin/env python3
"""Download PTB-XL 500Hz WFDB files from PhysioNet — parallel version.

Uses Python ThreadPoolExecutor for parallel curl downloads.
Resumes: skips files that already exist on disk.

Usage:
    python scripts/download_ptbxl_waveforms.py           # full download
    python scripts/download_ptbxl_waveforms.py --dry-run  # count only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"
LOCAL_DIR = Path(__file__).resolve().parent.parent / "data" / "ptbxl" / "wfdb"
SUBDIRS = [f"{i:05d}" for i in range(0, 22000, 1000)]
WORKERS = 16  # parallel curl processes
TIMEOUT = 30


def curl_fetch(url: str) -> str | None:
    """Fetch a URL with curl, return body as string or None on error."""
    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", str(TIMEOUT), url],
            capture_output=True, text=True, timeout=TIMEOUT + 5,
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return None


def list_remote_files(subdir: str) -> list[str]:
    """List .hea/.dat filenames in one remote subdirectory."""
    url = f"{BASE_URL}/records500/{subdir}/"
    html = curl_fetch(url)
    if html is None:
        print(f"  WARNING: failed to list {url}", file=sys.stderr)
        return []
    return re.findall(r'href="([^"]+\.(?:hea|dat))"', html)


def download_file(subdir: str, filename: str) -> tuple[str, bool, str]:
    """Download one file with curl. Returns (path_str, success, message)."""
    remote = f"{BASE_URL}/records500/{subdir}/{filename}"
    local = LOCAL_DIR / "records500" / subdir / filename
    local.parent.mkdir(parents=True, exist_ok=True)

    if local.exists() and local.stat().st_size > 0:
        return (str(local), True, "exists")

    try:
        result = subprocess.run(
            ["curl", "-sL", "--max-time", "60", "-o", str(local), remote],
            capture_output=True, timeout=65,
        )
        if result.returncode == 0 and local.exists() and local.stat().st_size > 0:
            return (str(local), True, f"{local.stat().st_size} bytes")
        else:
            if local.exists():
                local.unlink()
            return (str(local), False, f"curl exit {result.returncode}")
    except Exception as e:
        if local.exists():
            local.unlink()
        return (str(local), False, str(e))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Count files only")
    args = parser.parse_args()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: discover all remote files (sequential — just HTTP listing)
    print("Phase 1: Discovering remote files...", flush=True)
    all_files: list[tuple[str, str]] = []
    for subdir in SUBDIRS:
        files = list_remote_files(subdir)
        for f in files:
            all_files.append((subdir, f))
        print(f"  {subdir}: {len(files)} files", flush=True)

    total = len(all_files)
    print(f"\nTotal files to download: {total}", flush=True)

    if args.dry_run:
        return

    # Phase 2: parallel download
    print(f"\nPhase 2: Downloading {total} files with {WORKERS} workers...", flush=True)
    t0 = time.time()
    done = 0
    skipped = 0
    failed = 0
    failed_files: list[str] = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(download_file, sub, fname): (sub, fname)
            for sub, fname in all_files
        }
        for future in as_completed(futures):
            path_str, ok, msg = future.result()
            done += 1
            if msg == "exists":
                skipped += 1
            elif not ok:
                failed += 1
                failed_files.append(path_str)

            if done % 1000 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                pct = done / total * 100
                print(f"  [{done}/{total}] {pct:.1f}% | {rate:.0f} files/s | "
                      f"skipped={skipped} failed={failed}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s", flush=True)
    print(f"  Downloaded: {done - skipped - failed}", flush=True)
    print(f"  Skipped (already existed): {skipped}", flush=True)
    print(f"  Failed: {failed}", flush=True)

    if failed_files:
        print(f"\nFailed files (first 20):", flush=True)
        for f in failed_files[:20]:
            print(f"  {f}", flush=True)

    # Phase 3: verify count
    local_hea = sum(1 for _ in LOCAL_DIR.rglob("*.hea"))
    local_dat = sum(1 for _ in LOCAL_DIR.rglob("*.dat"))
    print(f"\nVerification:", flush=True)
    print(f"  .hea files on disk: {local_hea}", flush=True)
    print(f"  .dat files on disk: {local_dat}", flush=True)
    expected = total // 2
    if local_hea >= expected * 0.99:
        print(f"  PASS: {local_hea} .hea files >= 99% of expected {expected}", flush=True)
    else:
        print(f"  WARNING: only {local_hea}/{expected} .hea files present", flush=True)


if __name__ == "__main__":
    main()
