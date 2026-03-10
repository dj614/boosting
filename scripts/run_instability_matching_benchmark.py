#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from progress_utils import progress_bar, tqdm_iter
from sim.instability_matching_analysis import save_json, save_table
from sim.instability_matching_data import generate_dataset_bundle, summarize_dataset_bundle
from sim.instability_matching_eval import (
    aggregate_prediction_variance,
    compute_metrics,
    groupwise_prediction_variance,
    subgroup_metrics,
)
from sim.instability_matching_models import build_model, default_methods_for_task, make_default_learner_specs


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment 1 instability-matching benchmark.")
    parser.add_argument("--tasks", nargs="+", choices=["regression", "classification"], default=["regression", "classification"])
    parser.add_argument("--scenarios", nargs="+", choices=["piecewise", "smooth", "pocket"], default=["piecewise", "smooth", "pocket"])
    parser.add_argument("--methods", nargs="+", default=None, help="Optional explicit method list. Defaults to task-specific built-ins.")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=25)
    parser.add_argument("--eval-split", choices=["valid", "test"], default="test")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-valid", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--noise-type", choices=["homoscedastic", "heteroscedastic"], default="homoscedastic")
    parser.add_argument("--feature-dist", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--save-pointwise", action="store_true", help="Save pointwise mean predictions for downstream geometry plots.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment1_instability"))
    return parser


def _resolve_methods(task_type: str, requested_methods: List[str] | None) -> List[str]:
    if requested_methods is None:
        return default_methods_for_task(task_type)
    available = make_default_learner_specs(random_state=0)
    task_methods = [name for name in requested_methods if name in available and available[name].task_type == task_type]
    if not task_methods:
        raise ValueError(f"No requested methods are compatible with task_type={task_type!r}")
    return task_methods


def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n, endpoint=False)


def _force_single_thread(model: object) -> None:
    estimator = getattr(model, "estimator", model)
    if hasattr(estimator, "get_params") and "n_jobs" in estimator.get_params(deep=False):
        estimator.set_params(n_jobs=1)


def _split_from_bundle(bundle, split_name: str):
    if split_name == "valid":
        return bundle.valid
    return bundle.test


def _pointwise_frame(
    *,
    task_type: str,
    scenario: str,
    rep: int,
    method: str,
    split,
    mean_pred: np.ndarray,
) -> pd.DataFrame:
    frame_dict: Dict[str, np.ndarray | List[str] | List[int]] = {
        "task_type": [task_type] * split.X.shape[0],
        "scenario": [scenario] * split.X.shape[0],
        "rep": [rep] * split.X.shape[0],
        "method": [method] * split.X.shape[0],
        "test_index": np.arange(split.X.shape[0], dtype=int),
        "y_true": np.asarray(split.y),
        "f_true": np.asarray(split.f_true),
        "pred_mean": np.asarray(mean_pred, dtype=float),
    }
    for key, value in split.meta.items():
        if np.asarray(value).ndim == 1:
            frame_dict[key] = np.asarray(value)
    frame = pd.DataFrame(frame_dict)
    for feature_idx in range(min(3, split.X.shape[1])):
        frame[f"feature_{feature_idx}"] = split.X[:, feature_idx]
    return frame


def _trial_row(
    *,
    task_type: str,
    scenario: str,
    rep: int,
    method: str,
    bootstrap_reps: int,
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    pred_matrix: np.ndarray,
    meta: Dict[str, np.ndarray],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "task_type": task_type,
        "scenario": scenario,
        "rep": rep,
        "method": method,
        "bootstrap_reps": bootstrap_reps,
    }
    row.update(compute_metrics(task_type, y_true=y_true, pred=mean_pred))
    row.update(subgroup_metrics(task_type, y_true=y_true, pred=mean_pred, meta=meta))
    row.update(aggregate_prediction_variance(pred_matrix))
    row.update(groupwise_prediction_variance(pred_matrix, meta=meta))
    return row


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    config = {
        "tasks": args.tasks,
        "scenarios": args.scenarios,
        "methods": args.methods,
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "bootstrap_reps": args.bootstrap_reps,
        "eval_split": args.eval_split,
        "n_train": args.n_train,
        "n_valid": args.n_valid,
        "n_test": args.n_test,
        "p": args.p,
        "noise_type": args.noise_type,
        "feature_dist": args.feature_dist,
        "noise_scale": args.noise_scale,
        "save_pointwise": bool(args.save_pointwise),
    }
    save_json(config, outdir / "run_config.json")

    trial_rows: List[Dict[str, object]] = []
    pointwise_frames: List[pd.DataFrame] = []
    dataset_rows: List[Dict[str, object]] = []

    total_method_trials = sum(
        len(_resolve_methods(task_type, args.methods)) * len(args.scenarios) * args.num_seeds for task_type in args.tasks
    )

    with progress_bar(total=total_method_trials, desc="Experiment 1 benchmark", unit="model") as pbar:
        for task_type in args.tasks:
            methods = _resolve_methods(task_type, args.methods)
            for scenario in args.scenarios:
                for rep in range(args.num_seeds):
                    bundle = generate_dataset_bundle(
                        task_type=task_type,
                        scenario=scenario,
                        n_train=args.n_train,
                        n_valid=args.n_valid,
                        n_test=args.n_test,
                        p=args.p,
                        feature_dist=args.feature_dist,
                        noise_type=args.noise_type,
                        noise_scale=args.noise_scale,
                        seed=args.base_seed + rep,
                    )
                    split = _split_from_bundle(bundle, args.eval_split)
                    dataset_summary = summarize_dataset_bundle(bundle)
                    dataset_summary.update({"task_type": task_type, "scenario": scenario, "rep": rep})
                    dataset_rows.append(dataset_summary)

                    bootstrap_rng = np.random.default_rng(args.base_seed + 10000 * (rep + 1))
                    bootstrap_indices = [
                        _bootstrap_indices(bundle.train.X.shape[0], rng=bootstrap_rng) for _ in range(args.bootstrap_reps)
                    ]
                    for method in methods:
                        pbar.set_postfix(task=task_type, scenario=scenario, rep=rep, method=method)
                        pred_list: List[np.ndarray] = []
                        bootstrap_desc = f"{task_type}/{scenario}/rep{rep}/{method}"
                        for bootstrap_id, train_idx in enumerate(
                            tqdm_iter(
                                bootstrap_indices,
                                total=len(bootstrap_indices),
                                desc=bootstrap_desc,
                                unit="boot",
                                leave=False,
                            )
                        ):
                            model = build_model(
                                method,
                                task_type=task_type,
                                random_state=(args.base_seed + rep * 1000 + bootstrap_id),
                            )
                            _force_single_thread(model)
                            model.fit(bundle.train.X[train_idx], bundle.train.y[train_idx])
                            if task_type == "classification":
                                pred = model.predict_proba(split.X)
                            else:
                                pred = model.predict(split.X)
                            pred_list.append(np.asarray(pred, dtype=float).reshape(-1))

                        pred_matrix = np.vstack(pred_list)
                        mean_pred = pred_matrix.mean(axis=0)
                        trial_rows.append(
                            _trial_row(
                                task_type=task_type,
                                scenario=scenario,
                                rep=rep,
                                method=method,
                                bootstrap_reps=args.bootstrap_reps,
                                y_true=np.asarray(split.y),
                                mean_pred=mean_pred,
                                pred_matrix=pred_matrix,
                                meta=split.meta,
                            )
                        )
                        if args.save_pointwise:
                            pointwise_frames.append(
                                _pointwise_frame(
                                    task_type=task_type,
                                    scenario=scenario,
                                    rep=rep,
                                    method=method,
                                    split=split,
                                    mean_pred=mean_pred,
                                )
                            )
                        pbar.update(1)

    trial_df = pd.DataFrame(trial_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    save_table(trial_df, outdir / "trial_metrics.csv")
    save_table(dataset_df, outdir / "dataset_summaries.csv")

    if pointwise_frames:
        pointwise_df = pd.concat(pointwise_frames, ignore_index=True)
        save_table(pointwise_df, outdir / "pointwise_predictions.csv")

    quick_summary: Dict[str, object] = {
        "n_trials": float(len(trial_df)),
        "tasks": sorted(trial_df["task_type"].astype(str).unique().tolist()) if not trial_df.empty else [],
        "scenarios": sorted(trial_df["scenario"].astype(str).unique().tolist()) if not trial_df.empty else [],
        "methods": sorted(trial_df["method"].astype(str).unique().tolist()) if not trial_df.empty else [],
    }
    save_json(quick_summary, outdir / "artifact_summary.json")
    print(json.dumps(quick_summary, indent=2))
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
