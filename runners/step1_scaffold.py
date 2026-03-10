from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from data import generate_grouped_partial_linear_dataset, generate_heteroscedastic_regression_dataset
from metrics import evaluate_grouped_inference, evaluate_prediction_uq
from models import GroupedPartialLinearBaseline, RandomForestConformalRegressor
from plots.quicklook import save_grouped_inference_ci_plot, save_prediction_interval_width_plot

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for the step-1 scaffold runner") from exc


PREDICTION_BASELINES = {
    "rf_conformal": RandomForestConformalRegressor,
}

INFERENCE_BASELINES = {
    "grouped_partial_linear_baseline": GroupedPartialLinearBaseline,
}


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Top-level YAML config must decode to a mapping")
    return config


def save_yaml_config(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_step1_experiment(config: Dict[str, Any], output_root: Path) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    save_yaml_config(config, output_root / "config_snapshot.yaml")

    n_seeds = int(config.get("n_seeds", 1))
    alpha = float(config.get("alpha", 0.1))
    seeds = [int(config.get("seed_start", 0)) + i for i in range(n_seeds)]
    summary: Dict[str, object] = {"seeds": seeds}

    prediction_cfg = copy.deepcopy(config.get("prediction", {}))
    if prediction_cfg.get("enabled", True):
        summary["prediction"] = _run_prediction_track(prediction_cfg, output_root / "prediction", seeds, alpha)

    inference_cfg = copy.deepcopy(config.get("inference", {}))
    if inference_cfg.get("enabled", True):
        summary["inference"] = _run_inference_track(inference_cfg, output_root / "inference", seeds, alpha)

    with (output_root / "artifact_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _run_prediction_track(
    config: Dict[str, Any],
    output_dir: Path,
    seeds: List[int],
    alpha: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(config.get("model_name", "rf_conformal"))
    model_cls = PREDICTION_BASELINES[model_name]
    data_params = dict(config.get("data", {}))
    model_params = dict(config.get("model", {}))

    rows: List[Dict[str, float]] = []
    pointwise_rows: List[pd.DataFrame] = []

    for seed in seeds:
        dataset = generate_heteroscedastic_regression_dataset(seed=seed, **data_params)
        model = model_cls(alpha=alpha, random_state=seed, **model_params)
        model.fit(dataset)
        pred = model.predict(dataset.test.X)
        lower, upper = model.predict_interval(dataset.test.X, alpha=alpha)
        metrics = evaluate_prediction_uq(
            y_true=dataset.test.y,
            y_pred=pred,
            lower=lower,
            upper=upper,
            sigma_true=dataset.test.sigma_true,
        )
        rows.append({"seed": seed, **metrics})
        pointwise_rows.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "sample_id": dataset.test.sample_id,
                    "y_true": dataset.test.y,
                    "y_pred": pred,
                    "lower": lower,
                    "upper": upper,
                    "sigma_true": dataset.test.sigma_true,
                }
            )
        )

    per_seed = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary = _summarize_numeric_table(per_seed)
    per_seed.to_csv(output_dir / "per_seed_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.concat(pointwise_rows, ignore_index=True).to_csv(output_dir / "test_predictions.csv", index=False)
    save_prediction_interval_width_plot(pd.concat(pointwise_rows, ignore_index=True), output_dir / "prediction_interval_width.pdf")
    return {
        "model_name": model_name,
        "n_runs": len(rows),
        "summary_path": str(output_dir / "summary.csv"),
    }


def _run_inference_track(
    config: Dict[str, Any],
    output_dir: Path,
    seeds: List[int],
    alpha: float,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = str(config.get("model_name", "grouped_partial_linear_baseline"))
    model_cls = INFERENCE_BASELINES[model_name]
    data_params = dict(config.get("data", {}))
    model_params = dict(config.get("model", {}))

    rows: List[Dict[str, float]] = []

    for seed in seeds:
        dataset = generate_grouped_partial_linear_dataset(seed=seed, **data_params)
        model = model_cls(random_state=seed, **model_params)
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
        rows.append(
            {
                "seed": seed,
                "beta_hat": beta_hat,
                "beta_true": dataset.beta_true,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
                **metrics,
            }
        )

    per_seed = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary = _summarize_numeric_table(per_seed)
    per_seed.to_csv(output_dir / "per_seed_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_grouped_inference_ci_plot(per_seed, output_dir / "beta_ci_overview.pdf")
    return {
        "model_name": model_name,
        "n_runs": len(rows),
        "summary_path": str(output_dir / "summary.csv"),
    }


def _summarize_numeric_table(frame: pd.DataFrame) -> Dict[str, float]:
    numeric = frame.select_dtypes(include=["number"])
    return {f"{col}_mean": float(numeric[col].mean()) for col in numeric.columns if col != "seed"}
