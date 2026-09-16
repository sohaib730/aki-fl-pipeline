#!/usr/bin/env python3
"""
Reports and verifies the train/test split for both cohort master CSVs.

Usage:
    python3 record_train_test_numbers.py \
        aki_anchor_based_24h_lookback.csv \
        aki_anchor_based_24h_lookback_aligned_features.csv
"""
import sys
import pandas as pd

def report(path):
    df = pd.read_csv(path)
    n_total = len(df)
    n_subj = df['subject_id'].nunique()
    n_hadm = df['hadm_id'].nunique()

    print(f"=== {path} ===")
    print(f"  Total rows:            {n_total:,}")
    print(f"  Unique subject_id:     {n_subj:,}  {'(1:1 with rows -- OK)' if n_subj == n_total else '(MISMATCH -- investigate)'}")
    print(f"  Unique hadm_id:        {n_hadm:,}  {'(1:1 with rows -- OK)' if n_hadm == n_total else '(MISMATCH -- investigate)'}")

    if 'split' not in df.columns:
        print("  No 'split' column found -- cannot report train/test breakdown.")
        return None

    counts = df['split'].value_counts()
    n_train = counts.get('train', 0)
    n_test = counts.get('test', 0)
    print(f"  Train:                 {n_train:,}  ({n_train/n_total*100:.1f}%)")
    print(f"  Test:                  {n_test:,}  ({n_test/n_total*100:.1f}%)")

    train_ids = set(df[df['split'] == 'train']['subject_id'])
    test_ids = set(df[df['split'] == 'test']['subject_id'])
    overlap = train_ids & test_ids
    print(f"  Train/test subject_id overlap: {len(overlap):,}  {'(OK)' if len(overlap) == 0 else '(LEAKAGE -- investigate)'}")

    if 'AKI_label' in df.columns:
        print(f"  AKI prevalence (overall): {df['AKI_label'].mean()*100:.2f}%")

    return df[['subject_id', 'split']]

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 record_train_test_numbers.py <csv1> [csv2 ...]")
        sys.exit(1)

    results = []
    for path in sys.argv[1:]:
        r = report(path)
        results.append((path, r))
        print()

    # If exactly two files given, cross-check they share the same patients
    # and the same per-patient split assignment (as expected for Phase 1
    # vs Phase 2, which should be the identical underlying population).
    valid = [(p, r) for p, r in results if r is not None]
    if len(valid) == 2:
        (p1, r1), (p2, r2) = valid
        ids1, ids2 = set(r1['subject_id']), set(r2['subject_id'])
        print(f"=== Cross-file check: {p1} vs {p2} ===")
        print(f"  Same patient set: {ids1 == ids2}")
        merged = r1.merge(r2, on='subject_id', suffixes=('_1', '_2'))
        mismatches = merged[merged['split_1'] != merged['split_2']]
        print(f"  Split-assignment mismatches (same patient, different split label): {len(mismatches):,}")

if __name__ == "__main__":
    main()
