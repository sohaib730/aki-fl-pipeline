#!/usr/bin/env python3
"""
Checks cross-site patient overlap from the _subject_ids_*.csv files
produced by the patched simulation script (see HOW_TO_CHECK_OVERLAP.txt).

Disjoint sampling is only guaranteed WITHIN a single (alpha, gamma, seed)
condition -- each condition is an independent simulation run, so the same
patient legitimately can (and normally will) appear at "site_A under
alpha=0.1" and separately at "site_A under alpha=0.3". This script groups
files by condition (parsed from the filename) and checks overlap only
within each group, so pointing it at a full-grid output directory with
many conditions does not produce a false alarm.

Usage:
    python3 check_overlap.py ./overlap_check/
"""
import sys
import re
import glob
import itertools
import pandas as pd
from pathlib import Path
from collections import defaultdict

FNAME_RE = re.compile(r"^_subject_ids_(?P<site>.+)_alpha(?P<alpha>[\d.]+)_gamma(?P<gamma>[\d.]+)$")

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 check_overlap.py <output_dir>")
        sys.exit(1)

    out_dir = Path(sys.argv[1])
    files = sorted(glob.glob(str(out_dir / "_subject_ids_*.csv")))

    if not files:
        print(f"No _subject_ids_*.csv files found in {out_dir}")
        print("Did you apply the patch in HOW_TO_CHECK_OVERLAP.txt before running the simulation?")
        sys.exit(1)

    # Group files by (alpha, gamma) condition, parsed from the filename.
    # Files that don't match the expected naming pattern fall into a
    # single "_unknown_" bucket and are checked together as a fallback.
    conditions = defaultdict(dict)
    for f in files:
        stem = Path(f).stem
        m = FNAME_RE.match(stem)
        if m:
            site = m.group("site")
            cond = (m.group("alpha"), m.group("gamma"))
        else:
            site = stem.replace("_subject_ids_", "")
            cond = ("_unknown_", "_unknown_")
        conditions[cond][site] = set(pd.read_csv(f)["subject_id"])

    print(f"{len(files)} files loaded, grouped into {len(conditions)} condition(s) "
          f"(each checked for disjointness independently -- overlap ACROSS "
          f"different conditions is expected and not checked, since each "
          f"condition is a separate simulation run).\n")

    any_problem = False
    for (alpha, gamma), site_ids in sorted(conditions.items()):
        label = f"alpha={alpha} gamma={gamma}" if alpha != "_unknown_" else "(condition not parsed from filename)"
        print(f"=== {label} ({len(site_ids)} sites) ===")
        for name, ids in site_ids.items():
            print(f"  {name}: {len(ids):,} unique subject_ids")

        total_overlap_pairs = 0
        pair_lines = []
        for (a, ids_a), (b, ids_b) in itertools.combinations(site_ids.items(), 2):
            overlap = ids_a & ids_b
            if overlap:
                total_overlap_pairs += 1
                pct_a = 100 * len(overlap) / len(ids_a)
                pct_b = 100 * len(overlap) / len(ids_b)
                pair_lines.append(f"  {a} <-> {b}: {len(overlap):,} shared patients "
                                   f"({pct_a:.1f}% of {a}, {pct_b:.1f}% of {b})")

        if total_overlap_pairs == 0:
            print(f"  RESULT: no overlap across any of the "
                  f"{len(list(itertools.combinations(site_ids, 2)))} site pairs in this condition.\n")
        else:
            any_problem = True
            for line in pair_lines:
                print(line)
            all_ids = set.union(*site_ids.values())
            total_sampled = sum(len(v) for v in site_ids.values())
            print(f"  RESULT: overlap in {total_overlap_pairs} of "
                  f"{len(list(itertools.combinations(site_ids, 2)))} site pairs. "
                  f"{total_sampled - len(all_ids):,} of {total_sampled:,} draws "
                  f"({100*(total_sampled-len(all_ids))/total_sampled:.1f}%) are repeat "
                  f"appearances of a patient already used at another site "
                  f"WITHIN THIS SAME CONDITION.\n")

    print("=" * 60)
    if not any_problem:
        print(f"OVERALL: all {len(conditions)} condition(s) are internally disjoint "
              f"(no patient appears at more than one site within any single condition).")
    else:
        print(f"OVERALL: at least one condition has real cross-site overlap -- see above.")

if __name__ == "__main__":
    main()
