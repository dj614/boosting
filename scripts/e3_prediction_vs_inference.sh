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
if [[ ! -d "$REPO_DIR/data" ]]; then
  echo "[ERROR] REPO_DIR must point to the repo root containing data/, metrics/, models/, plots/. Got: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"
export REPO_DIR
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

export OUTDIR="${OUTDIR:-outputs/experiment3_prediction_vs_inference}"
export N_SEEDS="${N_SEEDS:-5}"
export SEED_START="${SEED_START:-0}"
export ALPHA="${ALPHA:-0.1}"

export PRED_N_TRAIN="${PRED_N_TRAIN:-600}"
export PRED_N_VALID="${PRED_N_VALID:-200}"
export PRED_N_CALIB="${PRED_N_CALIB:-200}"
export PRED_N_TEST="${PRED_N_TEST:-1000}"
export PRED_N_FEATURES="${PRED_N_FEATURES:-8}"
export PRED_BASE_NOISE="${PRED_BASE_NOISE:-0.75}"
export PRED_HETERO_STRENGTH="${PRED_HETERO_STRENGTH:-1.25}"
export PRED_N_ESTIMATORS="${PRED_N_ESTIMATORS:-200}"
export PRED_MAX_DEPTH="${PRED_MAX_DEPTH:-6}"
export PRED_MIN_SAMPLES_LEAF="${PRED_MIN_SAMPLES_LEAF:-5}"

export INF_N_GROUPS="${INF_N_GROUPS:-120}"
export INF_GROUP_SIZE="${INF_GROUP_SIZE:-8}"
export INF_N_FEATURES="${INF_N_FEATURES:-6}"
export INF_BETA_TRUE="${INF_BETA_TRUE:-1.0}"
export INF_GROUP_EFFECT_SCALE="${INF_GROUP_EFFECT_SCALE:-0.75}"
export INF_NOISE_SCALE="${INF_NOISE_SCALE:-0.75}"
export INF_WITHIN_GROUP_CORR="${INF_WITHIN_GROUP_CORR:-0.4}"
export INF_VALID_GROUP_FRAC="${INF_VALID_GROUP_FRAC:-0.2}"
export INF_TEST_GROUP_FRAC="${INF_TEST_GROUP_FRAC:-0.2}"
export INF_N_ESTIMATORS="${INF_N_ESTIMATORS:-200}"
export INF_MAX_DEPTH="${INF_MAX_DEPTH:-6}"
export INF_MIN_SAMPLES_LEAF="${INF_MIN_SAMPLES_LEAF:-5}"

echo "[e3] Running prediction and inference benchmark..."

python - <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

repo_dir = Path(os.environ["REPO_DIR"]).resolve()
outdir = repo_dir / os.environ["OUTDIR"]
outdir.mkdir(parents=True, exist_ok=True)

if str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

from data import generate_grouped_partial_linear_dataset, generate_heteroscedastic_regression_dataset
from metrics import evaluate_grouped_inference, evaluate_prediction_uq
from models.baselines import GroupedPartialLinearBaseline, RandomForestConformalRegressor
from plots.quicklook import save_grouped_inference_ci_plot, save_prediction_interval_width_plot

from progress_utils import tqdm_iter

def summarize_numeric_table(frame: pd.DataFrame) -> dict[str, float]:
    numeric = frame.select_dtypes(include=["number"])
    return {f"{col}_mean": float(numeric[col].mean()) for col in numeric.columns if col != "seed"}


n_seeds = int(os.environ["N_SEEDS"])
seed_start = int(os.environ["SEED_START"])
alpha = float(os.environ["ALPHA"])
seeds = [seed_start + i for i in range(n_seeds)]

config = {
    "n_seeds": n_seeds,
    "seed_start": seed_start,
    "alpha": alpha,
    "prediction": {
        "data": {
            "n_train": int(os.environ["PRED_N_TRAIN"]),
            "n_valid": int(os.environ["PRED_N_VALID"]),
            "n_calib": int(os.environ["PRED_N_CALIB"]),
            "n_test": int(os.environ["PRED_N_TEST"]),
            "n_features": int(os.environ["PRED_N_FEATURES"]),
            "base_noise": float(os.environ["PRED_BASE_NOISE"]),
            "hetero_strength": float(os.environ["PRED_HETERO_STRENGTH"]),
        },
        "model": {
            "n_estimators": int(os.environ["PRED_N_ESTIMATORS"]),
            "max_depth": int(os.environ["PRED_MAX_DEPTH"]),
            "min_samples_leaf": int(os.environ["PRED_MIN_SAMPLES_LEAF"]),
        },
    },
    "inference": {
        "data": {
            "n_groups": int(os.environ["INF_N_GROUPS"]),
            "group_size": int(os.environ["INF_GROUP_SIZE"]),
            "n_features": int(os.environ["INF_N_FEATURES"]),
            "beta_true": float(os.environ["INF_BETA_TRUE"]),
            "group_effect_scale": float(os.environ["INF_GROUP_EFFECT_SCALE"]),
            "noise_scale": float(os.environ["INF_NOISE_SCALE"]),
            "within_group_corr": float(os.environ["INF_WITHIN_GROUP_CORR"]),
            "valid_group_frac": float(os.environ["INF_VALID_GROUP_FRAC"]),
            "test_group_frac": float(os.environ["INF_TEST_GROUP_FRAC"]),
        },
        "model": {
            "n_estimators": int(os.environ["INF_N_ESTIMATORS"]),
            "max_depth": int(os.environ["INF_MAX_DEPTH"]),
            "min_samples_leaf": int(os.environ["INF_MIN_SAMPLES_LEAF"]),
        },
    },
}

with (outdir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(config, f, sort_keys=False)

# Prediction / uncertainty track
pred_out = outdir / "prediction"
pred_out.mkdir(parents=True, exist_ok=True)
pred_rows = []
pred_pointwise = []
for seed in tqdm_iter(seeds, total=len(seeds), desc="Experiment 3 prediction", unit="seed"):
    dataset = generate_heteroscedastic_regression_dataset(seed=seed, **config["prediction"]["data"])
    model = RandomForestConformalRegressor(alpha=alpha, random_state=seed, **config["prediction"]["model"])
    model.fit(dataset)
    y_pred = model.predict(dataset.test.X)
    lower, upper = model.predict_interval(dataset.test.X, alpha=alpha)
    metrics = evaluate_prediction_uq(
        y_true=dataset.test.y,
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        sigma_true=dataset.test.sigma_true,
    )
    pred_rows.append({"seed": seed, **metrics})
    pred_pointwise.append(
        pd.DataFrame(
            {
                "seed": seed,
                "sample_id": dataset.test.sample_id,
                "y_true": dataset.test.y,
                "y_pred": y_pred,
                "lower": lower,
                "upper": upper,
                "sigma_true": dataset.test.sigma_true,
            }
        )
    )

pred_per_seed = pd.DataFrame(pred_rows).sort_values("seed").reset_index(drop=True)
pred_summary = summarize_numeric_table(pred_per_seed)
pred_pointwise_df = pd.concat(pred_pointwise, ignore_index=True)
pred_per_seed.to_csv(pred_out / "per_seed_results.csv", index=False)
pd.DataFrame([pred_summary]).to_csv(pred_out / "summary.csv", index=False)
pred_pointwise_df.to_csv(pred_out / "test_predictions.csv", index=False)
with (pred_out / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(pred_summary, f, indent=2)
save_prediction_interval_width_plot(pred_pointwise_df, pred_out / "prediction_interval_width.pdf")

# Inference track
inf_out = outdir / "inference"
inf_out.mkdir(parents=True, exist_ok=True)
inf_rows = []
for seed in tqdm_iter(seeds, total=len(seeds), desc="Experiment 3 inference", unit="seed"):
    dataset = generate_grouped_partial_linear_dataset(seed=seed, **config["inference"]["data"])
    model = GroupedPartialLinearBaseline(random_state=seed, **config["inference"]["model"])
    model.fit(dataset)
    beta_hat = model.estimate_beta(dataset.test)
    beta_se = model.estimate_beta_se()
    ci = model.confidence_interval(alpha=alpha)
    y_pred = model.predict_mu(dataset.test.X) + beta_hat * dataset.test.D
    metrics = evaluate_grouped_inference(
        beta_hat=beta_hat,
        beta_true=dataset.beta_true,
        beta_se=beta_se,
        ci=ci,
        y_true=dataset.test.y,
        y_pred=y_pred,
    )
    inf_rows.append(
        {
            "seed": seed,
            "beta_hat": beta_hat,
            "beta_true": dataset.beta_true,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            **metrics,
        }
    )

inf_per_seed = pd.DataFrame(inf_rows).sort_values("seed").reset_index(drop=True)
inf_summary = summarize_numeric_table(inf_per_seed)
inf_per_seed.to_csv(inf_out / "per_seed_results.csv", index=False)
pd.DataFrame([inf_summary]).to_csv(inf_out / "summary.csv", index=False)
with (inf_out / "summary.json").open("w", encoding="utf-8") as f:
    json.dump(inf_summary, f, indent=2)
save_grouped_inference_ci_plot(inf_per_seed, inf_out / "beta_ci_overview.pdf")

summary = {
    "seeds": seeds,
    "prediction": {
        "model_name": "RandomForestConformalRegressor",
        "n_runs": len(pred_rows),
        "summary_path": str(pred_out / "summary.csv"),
    },
    "inference": {
        "model_name": "GroupedPartialLinearBaseline",
        "n_runs": len(inf_rows),
        "summary_path": str(inf_out / "summary.csv"),
    },
}
with (outdir / "artifact_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Wrote outputs to: {outdir}")
PY

if [[ "$OUTDIR" = /* ]]; then
  echo "Experiment 3 finished. Outputs: $OUTDIR"
else
  echo "Experiment 3 finished. Outputs: $REPO_DIR/$OUTDIR"
fi