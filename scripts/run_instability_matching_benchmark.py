#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from parallel_utils import make_process_pool, resolve_n_jobs
from progress_utils import progress_bar
from sim.instability_matching_analysis import save_json, save_table
from sim.instability_matching_data import generate_dataset_bundle, summarize_dataset_bundle
from sim.instability_matching_eval import (
    aggregate_prediction_variance,
    compute_metrics,
    groupwise_prediction_variance,
    subgroup_metrics,
)
from sim.ctb_semantics import canonical_ctb_tree_result_method, normalize_ctb_tree_method_name
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
    parser.add_argument("--ctb-n-estimators", type=int, default=50)
    parser.add_argument("--ctb-inner-bootstraps", type=int, default=8)
    parser.add_argument("--ctb-eta", type=float, default=1.0)
    parser.add_argument("--ctb-instability-penalty", type=float, default=0.0)
    parser.add_argument("--ctb-weight-power", type=float, default=1.0)
    parser.add_argument("--ctb-weight-eps", type=float, default=1e-8)
    parser.add_argument("--ctb-target-modes", nargs="*", default=["legacy"])
    parser.add_argument("--ctb-curvature-eps", nargs="*", type=float, default=[1e-6])
    parser.add_argument("--ctb-min-samples-leaf", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--save-pointwise", action="store_true", help="Save pointwise mean predictions for downstream geometry plots.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment1_instability"))
    return parser


def _resolve_methods(task_type: str, requested_methods: List[str] | None, spec_kwargs: Dict[str, object]) -> List[str]:
    if requested_methods is None:
        return default_methods_for_task(task_type)
    available = make_default_learner_specs(random_state=0, **spec_kwargs)
    task_methods: List[str] = []
    seen = set()
    for name in requested_methods:
        if name not in available or available[name].task_type != task_type:
            continue
        canonical_name = normalize_ctb_tree_method_name(name)
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        task_methods.append(str(name))
    if not task_methods:
        raise ValueError(f"No requested methods are compatible with task_type={task_type!r}")
    return task_methods


def _expand_method_specs(
    methods: List[str],
    *,
    ctb_target_modes: List[str],
    ctb_curvature_eps: List[float],
    base_spec_kwargs: Dict[str, object],
) -> List[Dict[str, object]]:
    expanded: List[Dict[str, object]] = []
    for method in methods:
        canonical_method = normalize_ctb_tree_method_name(method)
        if not canonical_method.startswith("ctb_"):
            expanded.append(
                {
                    "method": str(method),
                    "result_method": canonical_method,
                    "spec_kwargs": dict(base_spec_kwargs),
                }
            )
            continue
        for target_mode in ctb_target_modes:
            for curvature_eps in ctb_curvature_eps:
                spec_kwargs = dict(base_spec_kwargs)
                spec_kwargs.update(
                    {
                        "ctb_target_mode": str(target_mode),
                        "ctb_curvature_eps": float(curvature_eps),
                    }
                )
                expanded.append(
                    {
                        "method": str(method),
                        "result_method": canonical_ctb_tree_result_method(
                            canonical_method,
                            update_target_mode=str(target_mode),
                            transport_curvature_eps=float(curvature_eps),
                        ),
                        "spec_kwargs": spec_kwargs,
                    }
                )
    return expanded


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


def _run_method_trial(task: Dict[str, object]) -> Dict[str, object]:
    task_type = str(task["task_type"])
    scenario = str(task["scenario"])
    rep = int(task["rep"])
    base_seed = int(task["base_seed"])
    bootstrap_reps = int(task["bootstrap_reps"])
    spec_kwargs = dict(task["spec_kwargs"])

    bundle = generate_dataset_bundle(
        task_type=task_type,
        scenario=scenario,
        n_train=int(task["n_train"]),
        n_valid=int(task["n_valid"]),
        n_test=int(task["n_test"]),
        p=int(task["p"]),
        feature_dist=str(task["feature_dist"]),
        noise_type=str(task["noise_type"]),
        noise_scale=float(task["noise_scale"]),
        seed=base_seed + rep,
    )
    split = _split_from_bundle(bundle, str(task["eval_split"]))
    dataset_summary = summarize_dataset_bundle(bundle)
    dataset_summary.update({"task_type": task_type, "scenario": scenario, "rep": rep})

    bootstrap_rng = np.random.default_rng(base_seed + 10000 * (rep + 1))
    bootstrap_indices = [
        _bootstrap_indices(bundle.train.X.shape[0], rng=bootstrap_rng) for _ in range(bootstrap_reps)
    ]

    method = str(task["method"])
    result_method = str(task.get("result_method", method))
    pred_list: List[np.ndarray] = []
    for bootstrap_id, train_idx in enumerate(bootstrap_indices):
        model = build_model(
            method,
            task_type=task_type,
            random_state=(base_seed + rep * 1000 + bootstrap_id),
            **spec_kwargs,
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
    result: Dict[str, object] = {
        "trial_row": _trial_row(
            task_type=task_type,
            scenario=scenario,
            rep=rep,
            method=result_method,
            bootstrap_reps=bootstrap_reps,
            y_true=np.asarray(split.y),
            mean_pred=mean_pred,
            pred_matrix=pred_matrix,
            meta=split.meta,
        ),
        "dataset_summary": dataset_summary,
    }
    if bool(task["save_pointwise"]):
        result["pointwise_frame"] = _pointwise_frame(
            task_type=task_type,
            scenario=scenario,
            rep=rep,
            method=result_method,
            split=split,
            mean_pred=mean_pred,
        )
    return result


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
        "ctb_n_estimators": args.ctb_n_estimators,
        "ctb_inner_bootstraps": args.ctb_inner_bootstraps,
        "ctb_eta": args.ctb_eta,
        "ctb_instability_penalty": args.ctb_instability_penalty,
        "ctb_weight_power": args.ctb_weight_power,
        "ctb_weight_eps": args.ctb_weight_eps,
        "ctb_target_modes": list(args.ctb_target_modes),
        "ctb_curvature_eps": [float(x) for x in args.ctb_curvature_eps],
        "ctb_min_samples_leaf": args.ctb_min_samples_leaf,
        "save_pointwise": bool(args.save_pointwise),
    }
    save_json(config, outdir / "run_config.json")

    spec_kwargs: Dict[str, object] = {
        "ctb_n_estimators": int(args.ctb_n_estimators),
        "ctb_inner_bootstraps": int(args.ctb_inner_bootstraps),
        "ctb_eta": float(args.ctb_eta),
        "ctb_instability_penalty": float(args.ctb_instability_penalty),
        "ctb_weight_power": float(args.ctb_weight_power),
        "ctb_weight_eps": float(args.ctb_weight_eps),
        "ctb_target_mode": str(args.ctb_target_modes[0]),
        "ctb_curvature_eps": float(args.ctb_curvature_eps[0]),
        "ctb_min_samples_leaf": int(args.ctb_min_samples_leaf),
    }

    trial_rows: List[Dict[str, object]] = []
    pointwise_frames: List[pd.DataFrame] = []
    dataset_row_map: Dict[tuple[str, str, int], Dict[str, object]] = {}

    tasks: List[Dict[str, object]] = []
    for task_type in args.tasks:
        methods = _resolve_methods(task_type, args.methods, spec_kwargs)
        method_specs = _expand_method_specs(
            methods,
            ctb_target_modes=[str(x) for x in args.ctb_target_modes],
            ctb_curvature_eps=[float(x) for x in args.ctb_curvature_eps],
            base_spec_kwargs=spec_kwargs,
        )
        for scenario in args.scenarios:
            for rep in range(args.num_seeds):
                for method_spec in method_specs:
                    tasks.append(
                        {
                            "task_type": task_type,
                            "scenario": scenario,
                            "rep": rep,
                            "method": method_spec["method"],
                            "result_method": method_spec["result_method"],
                            "base_seed": int(args.base_seed),
                            "bootstrap_reps": int(args.bootstrap_reps),
                            "eval_split": args.eval_split,
                            "n_train": int(args.n_train),
                            "n_valid": int(args.n_valid),
                            "n_test": int(args.n_test),
                            "p": int(args.p),
                            "noise_type": args.noise_type,
                            "feature_dist": args.feature_dist,
                            "noise_scale": float(args.noise_scale),
                            "spec_kwargs": method_spec["spec_kwargs"],
                            "save_pointwise": bool(args.save_pointwise),
                        }
                    )

    n_jobs = resolve_n_jobs(args.n_jobs)
    with progress_bar(total=len(tasks), desc="Experiment 1 benchmark", unit="model") as pbar:
        if n_jobs <= 1:
            results = []
            for task in tasks:
                results.append(_run_method_trial(task))
                pbar.update(1)
        else:
            results = []
            with make_process_pool(n_jobs) as executor:
                futures = [executor.submit(_run_method_trial, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

    for result in results:
        trial_rows.append(result["trial_row"])
        dataset_summary = result["dataset_summary"]
        dataset_key = (
            str(dataset_summary["task_type"]),
            str(dataset_summary["scenario"]),
            int(dataset_summary["rep"]),
        )
        dataset_row_map.setdefault(dataset_key, dataset_summary)
        pointwise_frame = result.get("pointwise_frame")
        if isinstance(pointwise_frame, pd.DataFrame):
            pointwise_frames.append(pointwise_frame)

    trial_df = pd.DataFrame(trial_rows)
    dataset_df = pd.DataFrame(dataset_row_map.values())
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
