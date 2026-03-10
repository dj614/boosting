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
NUM_SEEDS="${NUM_SEEDS:-10}"
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

run_repo_script scripts/run_sparse_recovery_benchmark.py \
  --designs independent block_correlated strong_collinear \
  --models l2boost bagged_componentwise lasso xgb_tree \
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
  --save-feature-tables \
  --outdir "$OUTDIR"

echo "[e4] Benchmark complete. Starting analysis..."

run_repo_script scripts/analyze_sparse_recovery.py \
  --input-dir "$OUTDIR" \
  --outdir "$OUTDIR/analysis" \
  --top-features "$TOP_FEATURES"

echo "Experiment 4 finished. Outputs: $REPO_DIR/$OUTDIR"