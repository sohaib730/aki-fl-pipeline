#!/bin/bash
# Phase 2 (GPC-aligned) training, against the disjoint-sites-corrected
# ./phase2_data_disjoint/ (patients disjoint across sites; cohort updated
# after the KDIGO baseline-SCr/CKD-exclusion fix -- see run_complete.md).
#
# Covers run_complete.md Sections 3 (v2.3, 9 runs), 4 (v2.5, 9 runs), and
# 5 (4 baselines x 3 seeds x 3 conditions, 36 runs) = 54 runs total.
#
# NOT included: Sections 6 and 7 (taxonomy/clustering-fix and FedAdapt
# confirmatory re-runs) -- both were re-runs to confirm an in-place code
# fix hadn't changed already-reported results. Section 3 already uses the
# taxonomy-fixed script (fedadapt_train_approach2_v2_3_ftablation_taxtest_v2_
# leakage_fixed_improvement.py), and Section 5 already includes fedadapt as
# one of the 4 baseline methods -- so for a from-scratch run against new
# data, Sections 6/7 would just repeat Section 3/5's commands verbatim.
# Also NOT included: Section 2's local_epochs sweep (1 vs 3 vs 5) -- that
# was diagnostic work to establish local_epochs=3 as the v2.3 optimum,
# which is already a settled, confirmed choice baked into every command
# below; no need to re-sweep it.

set -e

DATA_DIR="./phase2_data_disjoint"
OUT_ROOT="./results_phase2_training"

CONDITIONS=("0.0 0.0" "0.5 0.75" "1.0 1.0")
SEEDS=(42 123 456)

check_data() {
  local ALPHA=$1
  local GAMMA=$2
  if ! ls "${DATA_DIR}"/sim_KUMC_alpha${ALPHA}_gamma${GAMMA}.csv >/dev/null 2>&1; then
    echo "[skip] missing site files for alpha=$ALPHA gamma=$GAMMA in $DATA_DIR"
    return 1
  fi
  return 0
}

# [RESUME] Skip any job whose output already exists from an earlier,
# interrupted run of this same script.
check_done() {
  local OUT_DIR=$1
  local METHOD=$2
  if [ -f "${OUT_DIR}${METHOD}/fl_gain_correlation.csv" ]; then
    return 0
  fi
  return 1
}

# ========================================================================
# Section 3 -- v2.3 (manual K=2), 20-group + weighted, local_epochs=3 -- 9 runs
# ========================================================================
for cond in "${CONDITIONS[@]}"; do
  read -r ALPHA GAMMA <<< "$cond"
  check_data "$ALPHA" "$GAMMA" || continue
  for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${OUT_ROOT}/a${ALPHA}_g${GAMMA}_seed${SEED}_20group_weighted_v2.3_K2_lepoch3/"
    if check_done "$OUT_DIR" "fedadaptproto"; then
      echo "[resume-skip] already completed: [Sec 3] v2.3 alpha=$ALPHA gamma=$GAMMA seed=$SEED"
      continue
    fi
    echo "=== [Sec 3] v2.3 alpha=$ALPHA gamma=$GAMMA seed=$SEED ==="
    python3 phase2_gpc_aligned_train_v23.py \
      --data_dir "$DATA_DIR" --alpha "$ALPHA" --gamma "$GAMMA" --seed "$SEED" --local_epochs 3 \
      --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
      --discriminator_target group --n_clusters 2 \
      --output_dir "$OUT_DIR"
  done
done

# ========================================================================
# Section 4 -- v2.5 (auto-K), 20-group + weighted, local_epochs=1 -- 9 runs
# ========================================================================
for cond in "${CONDITIONS[@]}"; do
  read -r ALPHA GAMMA <<< "$cond"
  check_data "$ALPHA" "$GAMMA" || continue
  for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${OUT_ROOT}/a${ALPHA}_g${GAMMA}_seed${SEED}_20group_weighted_v2.5_bestckpt_fix/"
    if check_done "$OUT_DIR" "fedadaptproto"; then
      echo "[resume-skip] already completed: [Sec 4] v2.5 alpha=$ALPHA gamma=$GAMMA seed=$SEED"
      continue
    fi
    echo "=== [Sec 4] v2.5 alpha=$ALPHA gamma=$GAMMA seed=$SEED ==="
    python3 phase2_gpc_aligned_train_v25.py \
      --data_dir "$DATA_DIR" --alpha "$ALPHA" --gamma "$GAMMA" --seed "$SEED" --local_epochs 1 \
      --method fedadaptproto --group_taxonomy phase4_20group --group_class_weighting \
      --discriminator_target group --auto_k --k_min 2 --k_max 5 \
      --output_dir "$OUT_DIR"
  done
done

# ========================================================================
# Section 5 -- 4 baselines (fedavg/fedprox/scaffold/fedadapt), local_epochs=3
# -- 36 runs
# ========================================================================
for METHOD in fedavg fedprox scaffold fedadapt; do
  for cond in "${CONDITIONS[@]}"; do
    read -r ALPHA GAMMA <<< "$cond"
    check_data "$ALPHA" "$GAMMA" || continue
    for SEED in "${SEEDS[@]}"; do
      OUT_DIR="${OUT_ROOT}/a${ALPHA}_g${GAMMA}_seed${SEED}_20group_weighted_lepoch3_${METHOD}/"
      if check_done "$OUT_DIR" "$METHOD"; then
        echo "[resume-skip] already completed: [Sec 5] $METHOD alpha=$ALPHA gamma=$GAMMA seed=$SEED"
        continue
      fi
      echo "=== [Sec 5] $METHOD alpha=$ALPHA gamma=$GAMMA seed=$SEED ==="
      python3 phase2_gpc_aligned_train_v23.py \
        --data_dir "$DATA_DIR" --alpha "$ALPHA" --gamma "$GAMMA" --seed "$SEED" --local_epochs 3 \
        --method "$METHOD" --group_taxonomy phase4_20group --group_class_weighting \
        --output_dir "$OUT_DIR"
    done
  done
done

echo "Done: Phase 2 training (9 + 9 + 36 = 54 runs)."
