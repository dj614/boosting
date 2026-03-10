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

OUTDIR="${OUTDIR:-outputs/experiment2_group_risk}"
DATASET="${DATASET:-simulated}"          # simulated or adult
GROUP_DEFINITION="${GROUP_DEFINITION:-sex_age}"
N_SAMPLES="${N_SAMPLES:-12000}"
N_FEATURES="${N_FEATURES:-8}"
VALID_SIZE="${VALID_SIZE:-0.20}"
TEST_SIZE="${TEST_SIZE:-0.20}"
SEED_START="${SEED_START:-0}"
NUM_SEEDS="${NUM_SEEDS:-5}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
MIN_SAMPLES_LEAF="${MIN_SAMPLES_LEAF:-5}"
SUBSAMPLE="${SUBSAMPLE:-1.0}"
COLSAMPLE_BYTREE="${COLSAMPLE_BYTREE:-0.9}"
TRAJECTORY_EVERY="${TRAJECTORY_EVERY:-10}"
TRAJ_SAMPLES_PER_GROUP="${TRAJ_SAMPLES_PER_GROUP:-8}"

run_repo_script scripts/run_group_risk_trajectory_benchmark.py \
  --dataset "$DATASET" \
  --group-definition "$GROUP_DEFINITION" \
  --n-samples "$N_SAMPLES" \
  --n-features "$N_FEATURES" \
  --valid-size "$VALID_SIZE" \
  --test-size "$TEST_SIZE" \
  --seed-start "$SEED_START" \
  --num-seeds "$NUM_SEEDS" \
  --families bagging rf gbdt xgb \
  --max-depths 1 3 5 \
  --ensemble-sizes 20 50 100 300 \
  --trajectory-every "$TRAJECTORY_EVERY" \
  --learning-rate "$LEARNING_RATE" \
  --min-samples-leaf "$MIN_SAMPLES_LEAF" \
  --subsample "$SUBSAMPLE" \
  --colsample-bytree "$COLSAMPLE_BYTREE" \
  --prediction-splits valid test \
  --trajectory-splits valid test \
  --trajectory-sample-count-per-group "$TRAJ_SAMPLES_PER_GROUP" \
  --outdir "$OUTDIR"

echo "[e2] Benchmark complete. Starting analysis..."
run_repo_script scripts/analyze_group_risk_redistribution.py \
  --input-dir "$OUTDIR" \
  --outdir "$OUTDIR/analysis" \
  --split test \
  --bootstrap-iters 200 \
  --baseline-top-frac 0.10
  
maybe_log_to_wandb "$OUTDIR" "experiment2" "experiment2-group-risk" "experiment2,group-risk"

echo "Experiment 2 finished. Outputs: $REPO_DIR/$OUTDIR"