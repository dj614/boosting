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

OUTDIR="${OUTDIR:-outputs/experiment1_instability}"
NUM_SEEDS="${NUM_SEEDS:-10}"
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

run_repo_script scripts/run_instability_matching_benchmark.py \
  --tasks regression classification \
  --scenarios piecewise smooth pocket \
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
  --save-pointwise \
  --outdir "$OUTDIR"

echo "[e1] Benchmark complete. Starting analysis..."
run_repo_script scripts/analyze_instability_matching.py \
  --input-dir "$OUTDIR" \
  --outdir "$OUTDIR/analysis"

echo "Experiment 1 finished. Outputs: $REPO_DIR/$OUTDIR"