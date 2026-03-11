#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/scripts" ]]; then
  DEFAULT_REPO_DIR="$SCRIPT_DIR"
elif [[ -d "$SCRIPT_DIR/boosting/scripts" ]]; then
  DEFAULT_REPO_DIR="$SCRIPT_DIR/boosting"
else
  DEFAULT_REPO_DIR="$(pwd)"
fi

REPO_DIR="${1:-${REPO_DIR:-$DEFAULT_REPO_DIR}}"
if [[ ! -d "$REPO_DIR/scripts" ]]; then
  echo "[ERROR] REPO_DIR must point to the repo root containing scripts/. Got: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

WANDB_ENABLE="${WANDB_ENABLE:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-boosting}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_NOTES="${WANDB_NOTES:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_MAX_TABLE_ROWS="${WANDB_MAX_TABLE_ROWS:-5000}"
WANDB_LOG_ARTIFACTS="${WANDB_LOG_ARTIFACTS:-1}"

maybe_log_to_wandb() {
  local output_dir="$1"
  local job_type="$2"
  local default_name="$3"
  local default_tags="$4"

  if [[ "$WANDB_ENABLE" != "1" ]]; then
    return 0
  fi

  local args=(
    scripts/log_outputs_to_wandb.py
    --output-dir "$output_dir"
    --project "$WANDB_PROJECT"
    --job-type "$job_type"
    --mode "$WANDB_MODE"
    --max-table-rows "$WANDB_MAX_TABLE_ROWS"
  )

  if [[ -n "$WANDB_ENTITY" ]]; then
    args+=(--entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_RUN_GROUP" ]]; then
    args+=(--group "$WANDB_RUN_GROUP")
  fi
  if [[ -n "$WANDB_NOTES" ]]; then
    args+=(--notes "$WANDB_NOTES")
  fi

  local run_name="$default_name"
  if [[ -n "$WANDB_NAME" ]]; then
    run_name="$WANDB_NAME"
  fi
  args+=(--name "$run_name")

  local tags="$default_tags"
  if [[ -n "$WANDB_TAGS" ]]; then
    tags="$WANDB_TAGS"
  fi
  if [[ -n "$tags" ]]; then
    args+=(--tags "$tags")
  fi
  if [[ "$WANDB_LOG_ARTIFACTS" == "1" ]]; then
    args+=(--log-artifact)
  fi

  echo "[wandb] Logging outputs from $output_dir ..."
  python "${args[@]}"
}

run_repo_script() {
  local target_script="$1"
  shift
  TARGET_SCRIPT="$target_script" REPO_DIR="$REPO_DIR" python - "$@" <<'PY'
import importlib.util
import os
import runpy
import sys
from pathlib import Path

repo_dir = Path(os.environ["REPO_DIR"]).resolve()
if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

alias_src = repo_dir / "sim" / "group_risk_redistribution_analysis.py"
if alias_src.exists() and "sim.experiment1_step3_analysis" not in sys.modules:
    spec = importlib.util.spec_from_file_location("sim.experiment1_step3_analysis", alias_src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

sys.argv = [os.environ["TARGET_SCRIPT"], *sys.argv[1:]]
runpy.run_path(str(repo_dir / os.environ["TARGET_SCRIPT"]), run_name="__main__")
PY
}

OUTDIR="${OUTDIR:-outputs/experiment1_instability}"
NUM_SEEDS="${NUM_SEEDS:-5}"
BASE_SEED="${BASE_SEED:-0}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-25}"
N_TRAIN="${N_TRAIN:-500}"
N_VALID="${N_VALID:-500}"
N_TEST="${N_TEST:-5000}"
P="${P:-20}"
NOISE_TYPE="${NOISE_TYPE:-homoscedastic}"
FEATURE_DIST="${FEATURE_DIST:-uniform}"
NOISE_SCALE="${NOISE_SCALE:-0.5}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
CTB_N_ESTIMATORS="${CTB_N_ESTIMATORS:-100}"
CTB_INNER_BOOTSTRAPS="${CTB_INNER_BOOTSTRAPS:-16}"
CTB_ETA="${CTB_ETA:-0.1}"
CTB_INSTABILITY_PENALTY="${CTB_INSTABILITY_PENALTY:-0.2}"
CTB_WEIGHT_POWER="${CTB_WEIGHT_POWER:-1.0}"
CTB_WEIGHT_EPS="${CTB_WEIGHT_EPS:-1e-8}"
CTB_MIN_SAMPLES_LEAF="${CTB_MIN_SAMPLES_LEAF:-5}"

run_repo_script scripts/run_instability_matching_benchmark.py \
  --tasks regression classification \
  --scenarios piecewise smooth pocket \
  --methods \
    bagging_tree_deep_regression bagging_tree_shallow_regression gbdt_stump_regression gbdt_depth3_regression ctb_stump_regression ctb_depth3_regression \
    bagging_tree_deep_classification bagging_tree_shallow_classification gbdt_stump_classification gbdt_depth3_classification ctb_stump_classification ctb_depth3_classification \
  --num-seeds "$NUM_SEEDS" \
  --base-seed "$BASE_SEED" \
  --bootstrap-reps "$BOOTSTRAP_REPS" \
  --eval-split "$EVAL_SPLIT" \
  --n-train "$N_TRAIN" \
  --n-valid "$N_VALID" \
  --n-test "$N_TEST" \
  --p "$P" \
  --noise-type "$NOISE_TYPE" \
  --feature-dist "$FEATURE_DIST" \
  --noise-scale "$NOISE_SCALE" \
  --ctb-n-estimators "$CTB_N_ESTIMATORS" \
  --ctb-inner-bootstraps "$CTB_INNER_BOOTSTRAPS" \
  --ctb-eta "$CTB_ETA" \
  --ctb-instability-penalty "$CTB_INSTABILITY_PENALTY" \
  --ctb-weight-power "$CTB_WEIGHT_POWER" \
  --ctb-weight-eps "$CTB_WEIGHT_EPS" \
  --ctb-min-samples-leaf "$CTB_MIN_SAMPLES_LEAF" \
  --save-pointwise \
  --outdir "$OUTDIR"

echo "[e1] Benchmark complete. Starting analysis..."
run_repo_script scripts/analyze_instability_matching.py \
  --input-dir "$OUTDIR" \
  --outdir "$OUTDIR/analysis"

maybe_log_to_wandb "$OUTDIR" "experiment1" "experiment1-instability" "experiment1,instability"

echo "Experiment 1 finished. Outputs: $REPO_DIR/$OUTDIR"