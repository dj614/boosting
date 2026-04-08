#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# -----------------------------------------------------------------------------
# Open tabular benchmark: classification + regression
#
# This script now keeps a single output folder per task type:
# - one for classification
# - one for regression
#
# We still run baselines and CTB separately so CTB can keep its larger custom
# grid without changing baseline grids, but both runs write into the same outdir.
# The second call appends/merges its summaries into the existing folder.
#
# To keep outputs lightweight, per-family outputs only retain key artifacts.
# In addition, the runner writes compact family x benchmark CSV tables for:
# - test_accuracy
# - test_balanced_accuracy
# -----------------------------------------------------------------------------

N_JOBS=12
MAX_ROUNDS=300
SELECTION_CHECKPOINTS=(100 200 300)

CLASSIFICATION_DATASETS=(
  blood
  climate
  credit
  diabetes
  german_numer
  qsar
  raisin
  titanic
)

REGRESSION_DATASETS=(
  california_housing
  concrete_compressive_strength
  superconductivity
  diamonds
)

BASELINE_FAMILIES=(bagging rf gbdt xgb)
CTB_FAMILIES=(ctb)

# CTB-only grid: 12 points total.
CTB_MAX_DEPTHS=(1 3 5)
CTB_MIN_SAMPLES_LEAFS=(1 5)
CTB_INNER_BOOTSTRAPS=(2 4)
CTB_ETAS=(1.0)
CTB_LEAF_RIDGES=(1.0)

OUTPUT_ROOT="outputs/open_tabular_benchmark"
CLASSIFICATION_OUT="${OUTPUT_ROOT}/classification__select-report__families-all__ctb-grid12"
REGRESSION_OUT="${OUTPUT_ROOT}/regression__select-report__families-all__ctb-grid12"

run_task() {
  local task_name="$1"
  local outdir="$2"
  shift 2

  local -a dataset_args=("$@")

  echo "[run] ${task_name}: baselines -> ${outdir}"
  python scripts/run_open_tabular_benchmark.py \
    "${dataset_args[@]}" \
    --n-jobs "$N_JOBS" \
    --outdir "$outdir" \
    --lightweight-output \
    --use-report-metric-for-selection \
    --max-rounds "$MAX_ROUNDS" \
    --selection-checkpoints "${SELECTION_CHECKPOINTS[@]}" \
    --families "${BASELINE_FAMILIES[@]}"

  echo "[run] ${task_name}: ctb -> ${outdir}"
  python scripts/run_open_tabular_benchmark.py \
    "${dataset_args[@]}" \
    --n-jobs "$N_JOBS" \
    --outdir "$outdir" \
    --append-output \
    --lightweight-output \
    --use-report-metric-for-selection \
    --max-rounds "$MAX_ROUNDS" \
    --selection-checkpoints "${SELECTION_CHECKPOINTS[@]}" \
    --families "${CTB_FAMILIES[@]}" \
    --max-depths "${CTB_MAX_DEPTHS[@]}" \
    --min-samples-leafs "${CTB_MIN_SAMPLES_LEAFS[@]}" \
    --ctb-inner-bootstraps "${CTB_INNER_BOOTSTRAPS[@]}" \
    --ctb-etas "${CTB_ETAS[@]}" \
    --ctb-leaf-ridges "${CTB_LEAF_RIDGES[@]}"

  echo "[analyze] ${task_name}: ${outdir}"
  python scripts/analyze_open_tabular_benchmark.py \
    --input-dir "$outdir" \
    --task-types "$task_name"
}

run_task \
  classification \
  "$CLASSIFICATION_OUT" \
  --classification-datasets "${CLASSIFICATION_DATASETS[@]}" \
  --regression-datasets

run_task \
  regression \
  "$REGRESSION_OUT" \
  --classification-datasets \
  --regression-datasets "${REGRESSION_DATASETS[@]}"

# # experiments
# bash scripts/e1_instability_matching.sh
# bash scripts/e2_group_risk_redistribution.sh
# bash scripts/e4_sparse_recovery.sh
# bash scripts/e3_prediction_vs_inference.sh
