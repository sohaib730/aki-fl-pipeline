#!/bin/bash

set -e

ALPHAS=(0.1 0.3 0.5 1.0 10.0)
GAMMAS=(0.0 0.5 0.75 1.0)
SEEDS=(42 123 456)

DATA_DIR="./phase1_data_disjoint"
TRAIN_SCRIPT="phase1_archetype_train_v25.py"

# Optional tag: ./run_phase1_grid_v25.sh postfix  ->  output goes to
# ./results_phase1_grid_v25_postfix/ instead of ./results_phase1_grid_v25/.
# Use this to keep runs from different pipeline states clearly separated
# on disk (e.g. "postfix" for a run against the corrected training script),
# rather than risking the mixed-provenance situation from before.
TAG="${1:-}"
if [ -n "$TAG" ]; then
  OUT_ROOT="./results_phase1_grid_v25_${TAG}"
else
  OUT_ROOT="./results_phase1_grid_v25"
fi

# Version safeguard: refuse to run unless the training script on disk is
# confirmed to be the fixed version (both the site-discovery filtering fix
# and the condition-specific cache-path fix). Prevents silently re-running
# against a stale/reverted copy of the script.
if ! grep -q "FIX.*original filter" "$TRAIN_SCRIPT" 2>/dev/null; then
  echo "ERROR: $TRAIN_SCRIPT does not contain the expected fix signature."
  echo "This script requires the fixed version (site-discovery filtering +"
  echo "condition-specific cache path). Refusing to run against what may be"
  echo "a stale or reverted copy -- verify the file before re-running."
  exit 1
fi

echo "Confirmed: $TRAIN_SCRIPT has the expected fix signature. Proceeding."
echo "Output directory: $OUT_ROOT"

for ALPHA in "${ALPHAS[@]}"; do
  for GAMMA in "${GAMMAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      OUT_DIR="${OUT_ROOT}/alpha${ALPHA}_gamma${GAMMA}_seed${SEED}/"

      if [ -f "${OUT_DIR}fedadaptproto/fl_gain_correlation.csv" ]; then
        echo "[resume-skip] already completed: alpha=$ALPHA gamma=$GAMMA seed=$SEED"
        continue
      fi

      if ! ls "${DATA_DIR}"/site_A_alpha${ALPHA}_gamma${GAMMA}.csv >/dev/null 2>&1; then
        echo "[skip] missing site files for alpha=$ALPHA gamma=$GAMMA in $DATA_DIR"
        continue
      fi

      echo "=== v2.5 alpha=$ALPHA gamma=$GAMMA seed=$SEED ==="
      python3 "$TRAIN_SCRIPT" \
        --data_dir "$DATA_DIR" \
        --method fedadaptproto --alpha "$ALPHA" --gamma "$GAMMA" --seed "$SEED" \
        --local_epochs 1 \
        --warmup_rounds 10 --early_stop_patience 0 \
        --auto_k --k_min 2 --k_max 5 --k_warmup_epochs 5 \
        --output_dir "$OUT_DIR"
    done
  done
done

echo "Done: v2.5 auto-K grid, $OUT_ROOT (20 conditions x 3 seeds = 60 jobs)."
