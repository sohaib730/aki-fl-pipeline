#!/bin/bash
# Generates simulated site data for both cohorts using the disjoint-sites
# simulation scripts (patients cannot appear at more than one site; see
# HOW_TO_CHECK_OVERLAP.txt / check_overlap.py for verification).
#
# IMPORTANT: this must run against the UPDATED master CSVs (post KDIGO
# baseline-SCr / CKD-exclusion fix, 114,720 patients total, 91,776 in the
# train split). TARGET_N_PER_SITE inside both scripts has already been
# recalibrated for this cohort size -- the highest values tested clean
# (zero shortfall/duplication warnings): 17,000/site x 5 sites for Phase 1,
# 14,000/site x 6 sites for Phase 2. Do NOT run this against the old,
# pre-fix CSVs (163,038 patients) or against these scripts if the cohort
# size changes again without first rechecking that constant.

set -e

PHASE1_CSV="aki_anchor_based_24h_lookback.csv"
PHASE2_CSV="aki_anchor_based_24h_lookback_aligned_features.csv"
PHASE1_SCRIPT="phase1_archetype_simulation.py"
PHASE2_SCRIPT="phase2_gpc_aligned_simulation.py"

# =========================================================================
# STEP 1 — single-condition smoke test (fast; confirms no shortfall/
# duplication warnings before committing to the full grid below)
# =========================================================================
echo "=== Phase 1 smoke test (alpha=0.3, gamma=0.75, seed=42) ==="
python3 "$PHASE1_SCRIPT" \
  --input "$PHASE1_CSV" --label AKI_label \
  --alpha 0.3 --gamma 0.75 --seed 42 \
  --output ./phase1_data_disjoint/

echo "=== Phase 2 smoke test (alpha=0.5, gamma=0.75, seed=42) ==="
python3 "$PHASE2_SCRIPT" \
  --input "$PHASE2_CSV" --label AKI_label \
  --alpha 0.5 --gamma 0.75 --seed 42 \
  --output ./phase2_data_disjoint/

echo
echo "=== Verifying zero cross-site overlap (smoke test outputs) ==="
python3 check_overlap.py ./phase1_data_disjoint/
python3 check_overlap.py ./phase2_data_disjoint/
echo
echo "If either check above reports any overlap, or if the console output"
echo "above showed any [disjoint-sampling] shortfall/duplication warnings,"
echo "STOP here and re-examine TARGET_N_PER_SITE before proceeding."
echo

# =========================================================================
# STEP 2 — full grid
# =========================================================================

echo "=== Phase 1 full 20-condition grid (5 alpha x 4 gamma) ==="
for ALPHA in 0.1 0.3 0.5 1.0 10.0; do
  for GAMMA in 0.0 0.5 0.75 1.0; do
    python3 "$PHASE1_SCRIPT" \
      --input "$PHASE1_CSV" --label AKI_label \
      --alpha "$ALPHA" --gamma "$GAMMA" --seed 42 \
      --output ./phase1_data_disjoint/
  done
done

echo "=== Phase 2 full 3-condition set ==="
for cond in "0.0 0.0" "0.5 0.75" "1.0 1.0"; do
  read -r ALPHA GAMMA <<< "$cond"
  python3 "$PHASE2_SCRIPT" \
    --input "$PHASE2_CSV" --label AKI_label \
    --alpha "$ALPHA" --gamma "$GAMMA" --seed 42 \
    --output ./phase2_data_disjoint/
done

echo
echo "=== Final overlap verification across the full grid ==="
python3 check_overlap.py ./phase1_data_disjoint/
python3 check_overlap.py ./phase2_data_disjoint/

echo
echo "Done. Site CSVs are in ./phase1_data_disjoint/ and ./phase2_data_disjoint/."
echo "Proceed to training (Sections 3/4/10/11 of run_complete.md) only after"
echo "both overlap checks above report zero overlap."
