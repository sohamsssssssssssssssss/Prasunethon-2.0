"""Phase 1 deliverable: prove the patient-independent split is leak-free.

Run:  python tests/test_split_integrity.py
Also runs under pytest (test_* functions).

Prints (never just asserts silently):
  - total records / unique patients / columns available
  - pairwise patient intersection sizes between train, val, test
  - per-split patient counts and record counts
  - per-split diagnostic superclass counts (NORM, MI, STTC, CD, HYP)
  - count of patients with multiple recordings across the WHOLE dataset
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import SUPERCLASSES, load_split  # noqa: E402

SPLIT_ORDER = ["train", "val", "test"]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def test_split_integrity() -> None:
    df, _ = load_split()

    section("1. DATASET OVERVIEW")
    print(f"total records          : {len(df)}")
    print(f"total unique patients  : {df['patient_id'].nunique()}")
    print(f"columns available      : {list(df.columns)}")

    section("2. SPLIT ASSIGNMENT (strat_fold 1-8=train, 9=val, 10=test)")
    print(df["split"].value_counts().to_string())

    section("3. PATIENT-INDEPENDENCE CHECK (unique patient_id per split)")
    patient_sets = {
        split: set(df.loc[df["split"] == split, "patient_id"]) for split in SPLIT_ORDER
    }

    print("\nPer-split patient counts:")
    for split in SPLIT_ORDER:
        print(f"  {split:6s}: {len(patient_sets[split]):6d} unique patients | "
              f"{(df['split'] == split).sum():6d} records")

    print("\nPairwise patient intersections (must be 0):")
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        inter = patient_sets[a] & patient_sets[b]
        print(f"  {a:5s} ∩ {b:5s} = {len(inter):6d} shared patients")
        assert len(inter) == 0, f"LEAK: {a} and {b} share {len(inter)} patients!"

    # Also: every patient must appear in exactly one split (no duplicates).
    all_patients = set().union(*patient_sets.values())
    assert len(all_patients) == df["patient_id"].nunique(), (
        "LEAK: union of split patient sets != total unique patients"
    )
    print(f"\n  union of all split patient sets = {len(all_patients)} "
          f"== total unique patients {df['patient_id'].nunique()}  [OK]")
    print("\nPATIENT-INDEPENDENT SPLIT VERIFIED: all pairwise intersections are 0.")

    section("4. CLASS DISTRIBUTION PER SPLIT (diagnostic superclasses)")
    # Multi-label: a record may count in >1 superclass (sum > records).
    print("(counts are per-record; multi-label records count in each class)\n")
    header = "  " + "".join(f"{s:>8s}" for s in SUPERCLASSES) + f"{'':>10s}records"
    print(header)
    for split in SPLIT_ORDER:
        sub = df[df["split"] == split]
        counts = Counter(
            sc for labels in sub["superclass"] for sc in labels
        )
        row = "  " + "".join(f"{counts.get(sc, 0):>8d}" for sc in SUPERCLASSES)
        row += f"{len(sub):>10d}"
        print(f"{split:6s}{row}")
    # Report how many records carry >1 superclass (multi-label prevalence).
    multi = df[df["superclass"].apply(len) > 1]
    print(f"\nrecords with >1 superclass (whole dataset): {len(multi)}")

    section("5. PATIENTS WITH MULTIPLE RECORDINGS (whole dataset)")
    per_patient = df.groupby("patient_id").size()
    multi_rec = per_patient[per_patient > 1]
    print(f"patients with 1 recording : {(per_patient == 1).sum()}")
    print(f"patients with >1 recording: {len(multi_rec)}")
    print(f"  of which: 2 recordings  : {(multi_rec == 2).sum()}")
    print(f"            3 recordings  : {(multi_rec == 3).sum()}")
    print(f"            >3 recordings : {(multi_rec > 3).sum()}")
    print(f"extra recordings beyond 1st: {(per_patient - 1).sum()} "
          f"(these are the records that would feed a drift layer)")

    section("DONE")
    print("split integrity: PASS (printed intersection sizes are all 0)")


if __name__ == "__main__":
    test_split_integrity()
