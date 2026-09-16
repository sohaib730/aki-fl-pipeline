#!/bin/bash

set -e

ALPHAS=(0.1 0.3 0.5 1.0 10.0)
GAMMAS=(0.0 0.5 0.75 1.0)
SEEDS=(42 123 456)
METHODS=(fedadaptproto fedavg fedprox scaffold fedadapt)

DATA_DIR="./phase1_data_disjoint"
TRAIN_SCRIPT="phase1_archetype_train_v23.py"

# Optional tag: ./run_phase1_grid_v23.sh postfix  ->  output goes to
# ./results_phase1_grid_v23_postfix/ instead of ./results_phase1_grid_v23/.
TAG="${1:-}"
if [ -n "$TAG" ]; then
  OUT_ROOT="./results_phase1_grid_v23_${TAG}"
else
  OUT_ROOT="./results_phase1_grid_v23"
fi

# Version safeguard: v2.3 never had the site-discovery/cache bugs v2.5 had,
# but confirm the exact-match, alpha/gamma-specific file selection this
# script depends on is actually present before running -- refuses to run
# against an unexpected or corrupted copy of the training script.
if ! grep -q 'alpha{alpha}_gamma{gamma}' "$TRAIN_SCRIPT" 2>/dev/null; then
  echo "ERROR: $TRAIN_SCRIPT does not contain the expected exact-match"
  echo "alpha/gamma file-selection logic this grid depends on. Refusing to"
  echo "run against what may be an unexpected or corrupted copy -- verify"
  echo "the file before re-running."
  exit 1
fi

echo "Confirmed: $TRAIN_SCRIPT has the expected file-selection logic. Proceeding."
echo "Output directory: $OUT_ROOT"

for METHOD in "${METHODS[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for GAMMA in "${GAMMAS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${OUT_ROOT}/alpha${ALPHA}_gamma${GAMMA}_seed${SEED}/"

        if [ -f "${OUT_DIR}${METHOD}/fl_gain_correlation.csv" ]; then
          echo "[resume-skip] already completed: method=$METHOD alpha=$ALPHA gamma=$GAMMA seed=$SEED"
          continue
        fi

        if ! ls "${DATA_DIR}"/site_A_alpha${ALPHA}_gamma${GAMMA}.csv >/dev/null 2>&1; then
          echo "[skip] missing site files for alpha=$ALPHA gamma=$GAMMA in $DATA_DIR"
          continue
        fi

        EXTRA_ARGS=""
        if [ "$METHOD" == "fedadaptproto" ]; then
          EXTRA_ARGS="--n_clusters 2"
        fi

        echo "=== method=$METHOD alpha=$ALPHA gamma=$GAMMA seed=$SEED ==="
        python3 "$TRAIN_SCRIPT" \
          --data_dir "$DATA_DIR" \
          --alpha "$ALPHA" --gamma "$GAMMA" --seed "$SEED" \
          --method "$METHOD" $EXTRA_ARGS \
          --embedding_dim 64 --hidden_dim 128 \
          --output_dir "$OUT_DIR"
      done
    done
  done
done

echo "Done: v2.3-based grid, $OUT_ROOT (5 methods x 20 conditions x 3 seeds = 300 jobs)."
