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
export N_JOBS="${N_JOBS:-24}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

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

OUTDIR="${OUTDIR:-outputs/experiment4_sparse_recovery}"
NUM_SEEDS="${NUM_SEEDS:-5}"
BASE_SEED="${BASE_SEED:-0}"
N_TRAIN="${N_TRAIN:-200}"
N_VALID="${N_VALID:-200}"
N_TEST="${N_TEST:-2000}"
P="${P:-2000}"
S="${S:-10}"
RHO="${RHO:-0.5}"
BLOCK_SIZE="${BLOCK_SIZE:-20}"
BETA_SCALE="${BETA_SCALE:-1.0}"
BETA_PATTERN="${BETA_PATTERN:-equal}"
SUPPORT_STRATEGY="${SUPPORT_STRATEGY:-spaced}"
SNR="${SNR:-4.0}"
XGB_SUPPORT_K="${XGB_SUPPORT_K:-10}"
TOP_FEATURES="${TOP_FEATURES:-25}"
CTB_MAX_STEPS="${CTB_MAX_STEPS:-300}"
CTB_INNER_BOOTSTRAPS="${CTB_INNER_BOOTSTRAPS:-16}"
CTB_ETA="${CTB_ETA:-0.1}"
CTB_RESIDUAL_WEIGHT_POWER="${CTB_RESIDUAL_WEIGHT_POWER:-1.0}"
CTB_RESIDUAL_WEIGHT_EPS="${CTB_RESIDUAL_WEIGHT_EPS:-1e-8}"
CTB_CONSENSUS_FREQUENCY_POWER="${CTB_CONSENSUS_FREQUENCY_POWER:-2.0}"
CTB_CONSENSUS_SIGN_POWER="${CTB_CONSENSUS_SIGN_POWER:-1.0}"
CTB_INSTABILITY_LAMBDA="${CTB_INSTABILITY_LAMBDA:-1.0}"
CTB_INSTABILITY_POWER="${CTB_INSTABILITY_POWER:-1.0}"
CTB_MIN_CONSENSUS_FREQUENCY="${CTB_MIN_CONSENSUS_FREQUENCY:-0.25}"
CTB_MIN_SIGN_CONSISTENCY="${CTB_MIN_SIGN_CONSISTENCY:-0.75}"
CTB_SUPPORT_FREQUENCY_THRESHOLD="${CTB_SUPPORT_FREQUENCY_THRESHOLD:-0.05}"

run_repo_script scripts/run_sparse_recovery_benchmark.py \
  --designs independent block_correlated strong_collinear \
  --models l2boost bagged_componentwise ctb_sparse lasso xgb_tree \
  --num-seeds "$NUM_SEEDS" \
  --base-seed "$BASE_SEED" \
  --n-train "$N_TRAIN" \
  --n-valid "$N_VALID" \
  --n-test "$N_TEST" \
  --p "$P" \
  --s "$S" \
  --rho "$RHO" \
  --block-size "$BLOCK_SIZE" \
  --beta-scale "$BETA_SCALE" \
  --beta-pattern "$BETA_PATTERN" \
  --support-strategy "$SUPPORT_STRATEGY" \
  --snr "$SNR" \
  --xgb-support-k "$XGB_SUPPORT_K" \
  --ctb-max-steps "$CTB_MAX_STEPS" \
  --ctb-inner-bootstraps "$CTB_INNER_BOOTSTRAPS" \
  --ctb-eta "$CTB_ETA" \
  --ctb-residual-weight-power "$CTB_RESIDUAL_WEIGHT_POWER" \
  --ctb-residual-weight-eps "$CTB_RESIDUAL_WEIGHT_EPS" \
  --ctb-consensus-frequency-power "$CTB_CONSENSUS_FREQUENCY_POWER" \
  --ctb-consensus-sign-power "$CTB_CONSENSUS_SIGN_POWER" \
  --ctb-instability-lambda "$CTB_INSTABILITY_LAMBDA" \
  --ctb-instability-power "$CTB_INSTABILITY_POWER" \
  --ctb-min-consensus-frequency "$CTB_MIN_CONSENSUS_FREQUENCY" \
  --ctb-min-sign-consistency "$CTB_MIN_SIGN_CONSISTENCY" \
  --ctb-support-frequency-threshold "$CTB_SUPPORT_FREQUENCY_THRESHOLD" \
  --n-jobs "$N_JOBS" \
  --save-feature-tables \
  --outdir "$OUTDIR"

echo "[e4] Benchmark complete. Starting analysis..."

run_repo_script scripts/analyze_sparse_recovery.py \
  --input-dir "$OUTDIR" \
  --outdir "$OUTDIR/analysis" \
  --top-features "$TOP_FEATURES"

maybe_log_to_wandb "$OUTDIR" "experiment4" "experiment4-sparse-recovery" "experiment4,sparse-recovery"

echo "Experiment 4 finished. Outputs: $REPO_DIR/$OUTDIR"