#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sim.experiment1_data import DatasetBundle, generate_dataset_bundle
from sim.experiment1_eval import (
    aggregate_prediction_variance,
    compute_metrics,
    groupwise_prediction_variance,
    subgroup_metrics,
)
from sim.experiment1_models import default_methods_for_task, build_model


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment 1 step-2 training/evaluation loop.")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--scenario", choices=["piecewise", "smooth", "pocket"], default="piecewise")
    parser.add_argument("--noise-type", choices=["homoscedastic", "heteroscedastic"], default="homoscedastic")
    parser.add_argument("--feature-dist", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-valid", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--bootstrap-reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional explicit method list. Defaults depend on task type.",
    )
    parser.add_argument(
        "--save-pointwise-preds",
        action="store_true",
        help="If set, save one row per test point / repetition / bootstrap replicate.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment1_step2"))
    return parser


def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def _prediction_for_task(model, task_type: str, X: np.ndarray) -> np.ndarray:
    if task_type == "classification":
        return model.predict_proba(X)
    return model.predict(X)


def _run_single_fit(
    bundle: DatasetBundle,
    method_name: str,
    task_type: str,
    rng: np.random.Generator,
    bootstrap: bool,
    fit_seed: int,
) -> np.ndarray:
    X_train = bundle.train.X
    y_train = bundle.train.y
    if bootstrap:
        idx = _bootstrap_indices(X_train.shape[0], rng=rng)
        X_fit = X_train[idx]
        y_fit = y_train[idx]
    else:
        X_fit = X_train
        y_fit = y_train

    model = build_model(method_name=method_name, task_type=task_type, random_state=fit_seed)
    model.fit(X_fit, y_fit)
    return _prediction_for_task(model, task_type=task_type, X=bundle.test.X)


def _bundle_config(args: argparse.Namespace, rep_seed: int) -> Dict[str, object]:
    return {
        "task_type": args.task,
        "scenario": args.scenario,
        "n_train": args.n_train,
        "n_valid": args.n_valid,
        "n_test": args.n_test,
        "p": args.p,
        "feature_dist": args.feature_dist,
        "noise_type": args.noise_type,
        "noise_scale": args.noise_scale,
        "seed": rep_seed,
    }


def _oracle_frame(bundle: DatasetBundle) -> pd.DataFrame:
    data = {
        "test_index": np.arange(bundle.test.X.shape[0], dtype=int),
        "y_true": bundle.test.y,
        "f_true": bundle.test.f_true,
    }
    max_feature_cols = min(3, bundle.test.X.shape[1])
    for feat_idx in range(max_feature_cols):
        data[f"feature_{feat_idx}"] = bundle.test.X[:, feat_idx]
    for key, value in bundle.test.meta.items():
        data[key] = value
    return pd.DataFrame(data)


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    methods = args.methods or default_methods_for_task(args.task)
    config = vars(args).copy()
    (outdir / "run_config.json").write_text(json.dumps(config, indent=2, default=str))

    trial_rows: List[Dict[str, float]] = []
    pointwise_rows: List[pd.DataFrame] = []

    for rep in range(args.repetitions):
        rep_seed = args.seed + rep
        bundle = generate_dataset_bundle(**_bundle_config(args, rep_seed))
        test_meta = bundle.test.meta
        oracle_df = _oracle_frame(bundle)

        for method_idx, method_name in enumerate(methods):
            preds = []
            n_fits = max(1, args.bootstrap_reps)
            for boot in range(n_fits):
                fit_seed = args.seed + 1000 * rep + 17 * method_idx + boot
                pred = _run_single_fit(
                    bundle=bundle,
                    method_name=method_name,
                    task_type=args.task,
                    rng=np.random.default_rng(fit_seed + 12345),
                    bootstrap=args.bootstrap_reps > 1,
                    fit_seed=fit_seed,
                )
                preds.append(pred)
                if args.save_pointwise_preds:
                    pred_col = "pred_score" if args.task == "regression" else "pred_prob"
                    df = oracle_df.copy()
                    df["rep"] = rep
                    df["bootstrap_id"] = boot
                    df["method"] = method_name
                    df[pred_col] = pred
                    pointwise_rows.append(df)

            pred_mat = np.stack(preds, axis=0)
            pred_mean = pred_mat.mean(axis=0)
            row: Dict[str, float] = {
                "rep": float(rep),
                "method": method_name,
                "bootstrap_reps": float(n_fits),
            }
            row.update(compute_metrics(args.task, bundle.test.y, pred_mean))
            row.update(subgroup_metrics(args.task, bundle.test.y, pred_mean, test_meta))
            row.update(aggregate_prediction_variance(pred_mat))
            row.update(groupwise_prediction_variance(pred_mat, test_meta))
            trial_rows.append(row)

    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(outdir / "trial_summary.csv", index=False)

    grouped = trial_df.groupby("method", dropna=False)
    mean_df = grouped.mean(numeric_only=True)
    se_df = grouped.sem(numeric_only=True).add_suffix("_se")
    mean_df.join(se_df).to_csv(outdir / "method_summary.csv")

    if args.save_pointwise_preds and pointwise_rows:
        pd.concat(pointwise_rows, ignore_index=True).to_parquet(outdir / "pointwise_predictions.parquet", index=False)

    print(f"Wrote trial summary to: {outdir / 'trial_summary.csv'}")
    print(f"Wrote method summary to: {outdir / 'method_summary.csv'}")
    if args.save_pointwise_preds and pointwise_rows:
        print(f"Wrote pointwise predictions to: {outdir / 'pointwise_predictions.parquet'}")


if __name__ == "__main__":
    main()
