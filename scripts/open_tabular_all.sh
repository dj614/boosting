#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# -----------------------------------------------------------------------------
# Open tabular benchmark: classification + regression
#
# Design choices for this script:
# 1) run the non-CTB baselines as well, not only CTB;
# 2) keep the existing baseline grids unchanged;
# 3) run CTB in a separate call so we can enlarge only the CTB grid without
#    accidentally changing the other families' grids;
# 4) use clearer output directory names;
# 5) merge baseline + CTB summaries into one comparison directory for analysis.
#
# Grid-point budget note:
# - default bagging / rf grid size = 3 depths x 2 min_samples_leaf = 6 points
# - default gbdt / xgb grid size = 3 depths x 2 learning_rates x 2 subsamples = 12 points
# - CTB grid below is set to 12 points exactly:
#       3 depths x 2 min_samples_leaf x 2 inner_bootstraps = 12
#   so CTB stays around the 100%-120% grid-point budget of the strongest
#   baseline families, while preserving the baseline grids.
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
ALL_FAMILIES=(bagging rf gbdt xgb ctb)

# CTB-only grid: 12 points total.
CTB_MAX_DEPTHS=(1 3 5)
CTB_MIN_SAMPLES_LEAFS=(1 5)
CTB_INNER_BOOTSTRAPS=(2 4)
CTB_ETAS=(1.0)
CTB_LEAF_RIDGES=(1.0)

OUTPUT_ROOT="outputs/open_tabular_benchmark"

merge_open_tabular_outputs() {
  local merged_dir="$1"
  shift

  python - "$merged_dir" "$@" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

merged_dir = Path(sys.argv[1])
input_dirs = [Path(p) for p in sys.argv[2:]]
merged_dir.mkdir(parents=True, exist_ok=True)


def _concat_csv(filename: str) -> pd.DataFrame:
    frames = []
    for input_dir in input_dirs:
        path = input_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=0, ignore_index=True)
    preferred_cols = [
        "task_type",
        "dataset_name",
        "repeat_id",
        "family",
        "selected_checkpoint",
        "valid_selection_metric",
    ]
    ordered = [c for c in preferred_cols if c in out.columns]
    ordered += [c for c in out.columns if c not in ordered]
    return out.loc[:, ordered]

summary_test = _concat_csv("summary_test_metrics.csv")
summary_valid = _concat_csv("summary_valid_selection.csv")
errors = _concat_csv("errors.csv")

summary_test.to_csv(merged_dir / "summary_test_metrics.csv", index=False)
summary_valid.to_csv(merged_dir / "summary_valid_selection.csv", index=False)
if not errors.empty:
    errors.to_csv(merged_dir / "errors.csv", index=False)

payload = {
    "merged_from": [str(p) for p in input_dirs],
    "summary_test_metrics_path": str(merged_dir / "summary_test_metrics.csv"),
    "summary_valid_selection_path": str(merged_dir / "summary_valid_selection.csv"),
}
if not errors.empty:
    payload["errors_path"] = str(merged_dir / "errors.csv")
(merged_dir / "artifact_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

run_task() {
  local task_name="$1"
  local baselines_out="$2"
  local ctb_out="$3"
  local merged_out="$4"
  shift 4

  local -a dataset_args=("$@")

  echo "[run] ${task_name}: baselines -> ${baselines_out}"
  python scripts/run_open_tabular_benchmark.py \
    "${dataset_args[@]}" \
    --n-jobs "$N_JOBS" \
    --outdir "$baselines_out" \
    --use-report-metric-for-selection \
    --max-rounds "$MAX_ROUNDS" \
    --selection-checkpoints "${SELECTION_CHECKPOINTS[@]}" \
    --families "${BASELINE_FAMILIES[@]}"

  echo "[run] ${task_name}: ctb -> ${ctb_out}"
  python scripts/run_open_tabular_benchmark.py \
    "${dataset_args[@]}" \
    --n-jobs "$N_JOBS" \
    --outdir "$ctb_out" \
    --use-report-metric-for-selection \
    --max-rounds "$MAX_ROUNDS" \
    --selection-checkpoints "${SELECTION_CHECKPOINTS[@]}" \
    --families "${CTB_FAMILIES[@]}" \
    --max-depths "${CTB_MAX_DEPTHS[@]}" \
    --min-samples-leafs "${CTB_MIN_SAMPLES_LEAFS[@]}" \
    --ctb-inner-bootstraps "${CTB_INNER_BOOTSTRAPS[@]}" \
    --ctb-etas "${CTB_ETAS[@]}" \
    --ctb-leaf-ridges "${CTB_LEAF_RIDGES[@]}"

  echo "[merge] ${task_name}: ${merged_out}"
  merge_open_tabular_outputs "$merged_out" "$baselines_out" "$ctb_out"

  echo "[analyze] ${task_name}: ${merged_out}"
  python scripts/analyze_open_tabular_benchmark.py \
    --input-dir "$merged_out" \
    --task-types "$task_name"
}

CLASSIFICATION_BASELINES_OUT="${OUTPUT_ROOT}/classification__select-report__families-baselines-default"
CLASSIFICATION_CTB_OUT="${OUTPUT_ROOT}/classification__select-report__family-ctb__grid12_depth-1-3-5_leaf-1-5_innerboot-2-4"
CLASSIFICATION_MERGED_OUT="${OUTPUT_ROOT}/classification__select-report__families-all__baselines-default_plus_ctb-grid12"

REGRESSION_BASELINES_OUT="${OUTPUT_ROOT}/regression__select-report__families-baselines-default"
REGRESSION_CTB_OUT="${OUTPUT_ROOT}/regression__select-report__family-ctb__grid12_depth-1-3-5_leaf-1-5_innerboot-2-4"
REGRESSION_MERGED_OUT="${OUTPUT_ROOT}/regression__select-report__families-all__baselines-default_plus_ctb-grid12"

run_task \
  classification \
  "$CLASSIFICATION_BASELINES_OUT" \
  "$CLASSIFICATION_CTB_OUT" \
  "$CLASSIFICATION_MERGED_OUT" \
  --classification-datasets "${CLASSIFICATION_DATASETS[@]}" \
  --regression-datasets

run_task \
  regression \
  "$REGRESSION_BASELINES_OUT" \
  "$REGRESSION_CTB_OUT" \
  "$REGRESSION_MERGED_OUT" \
  --classification-datasets \
  --regression-datasets "${REGRESSION_DATASETS[@]}"

# # experiments
# bash scripts/e1_instability_matching.sh
# bash scripts/e2_group_risk_redistribution.sh
# bash scripts/e4_sparse_recovery.sh
# bash scripts/e3_prediction_vs_inference.sh
