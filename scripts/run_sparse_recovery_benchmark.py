#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

alias_src = ROOT / "sim" / "group_risk_redistribution_analysis.py"
if alias_src.exists() and "sim.experiment1_step3_analysis" not in sys.modules:
    spec = importlib.util.spec_from_file_location("sim.experiment1_step3_analysis", alias_src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

import numpy as np
import pandas as pd
from parallel_utils import make_process_pool, resolve_n_jobs
from progress_utils import progress_bar
from sim.sparse_recovery_data import generate_sparse_regression_dataset, summarize_sparse_regression_dataset
from sim.sparse_recovery_eval import make_feature_support_frame, regression_metrics, support_recovery_metrics
from sim.ctb_semantics import ctb_tree_model_name
from sim.sparse_recovery_models import build_experiment4_model


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment 4 sparse recovery benchmark.")
    parser.add_argument(
        "--designs",
        nargs="+",
        choices=["independent", "block_correlated", "strong_collinear"],
        default=["independent", "block_correlated", "strong_collinear"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["l2boost", "bagged_componentwise", "ctb_sparse", "ctb_tree", "lasso", "xgb_tree"],
        default=["l2boost", "bagged_componentwise", "ctb_sparse", "lasso", "xgb_tree"],
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-valid", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--p", type=int, default=2000)
    parser.add_argument("--s", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--beta-pattern", choices=["equal", "decay", "mixed_sign"], default="equal")
    parser.add_argument("--support-strategy", choices=["first", "spaced", "random"], default="spaced")
    parser.add_argument("--snr", type=float, default=4.0)
    parser.add_argument("--xgb-support-k", type=int, default=None, help="If set, use top-k feature importance as XGBoost support estimate.")
    parser.add_argument("--ctb-max-steps", type=int, default=300)
    parser.add_argument("--ctb-inner-bootstraps", type=int, default=8)
    parser.add_argument("--ctb-eta", type=float, default=1.0)
    parser.add_argument("--ctb-residual-weight-power", type=float, default=1.0)
    parser.add_argument("--ctb-residual-weight-eps", type=float, default=1e-8)
    parser.add_argument("--ctb-consensus-frequency-power", type=float, default=2.0)
    parser.add_argument("--ctb-consensus-sign-power", type=float, default=1.0)
    parser.add_argument("--ctb-instability-lambda", type=float, default=1.0)
    parser.add_argument("--ctb-instability-power", type=float, default=1.0)
    parser.add_argument("--ctb-min-consensus-frequency", type=float, default=0.25)
    parser.add_argument("--ctb-min-sign-consistency", type=float, default=0.75)
    parser.add_argument("--ctb-support-frequency-threshold", type=float, default=0.05)
    parser.add_argument("--ctb-tree-max-depths", nargs="*", type=int, default=[1, 3])
    parser.add_argument("--ctb-tree-min-samples-leaf", type=int, default=5)
    parser.add_argument("--ctb-target-modes", nargs="*", default=["legacy"])
    parser.add_argument("--ctb-curvature-eps", nargs="*", type=float, default=[1e-6])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--save-feature-tables", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment4_sparse_recovery"))
    return parser


def _save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _resolve_support_hat(model_family: str, model: object, dataset, xgb_support_k: int | None) -> tuple[np.ndarray, str]:
    if model_family in {"xgb_tree", "ctb_tree"}:
        k = int(xgb_support_k or dataset.support_true.shape[0])
        support_hat = np.asarray(model.topk_support(k), dtype=int)
        return support_hat, f"topk_importance@{k}"
    support_hat = np.asarray(getattr(model, "selected_support_", np.array([], dtype=int)), dtype=int)
    return support_hat, "native"


def _feature_frame_for_model(model_family: str, model: object, dataset, support_hat: Sequence[int], support_mode: str) -> pd.DataFrame:
    p = dataset.train.X.shape[1]
    feature_names = dataset.feature_names
    extra_columns = {
        "corr_to_signal": dataset.train.meta["corr_to_signal"],
        "true_beta": dataset.beta_true,
    }

    if model_family in {"bagged_componentwise", "ctb_sparse"} and getattr(model, "selection_frequency_", None) is not None:
        frame = make_feature_support_frame(
            selection_frequency=np.asarray(model.selection_frequency_, dtype=float),
            support_true=dataset.support_true,
            feature_names=feature_names,
            extra_columns=extra_columns,
        )
    elif model_family in {"xgb_tree", "ctb_tree"} and getattr(model, "feature_importances_", None) is not None:
        frame = make_feature_support_frame(
            selection_frequency=np.asarray(model.feature_importances_, dtype=float),
            support_true=dataset.support_true,
            feature_names=feature_names,
            extra_columns=extra_columns,
        )
    else:
        indicator = np.zeros(p, dtype=float)
        indicator[np.asarray(list(support_hat), dtype=int)] = 1.0
        frame = make_feature_support_frame(
            selection_frequency=indicator,
            support_true=dataset.support_true,
            feature_names=feature_names,
            extra_columns=extra_columns,
        )

    frame["selected_for_support_eval"] = 0
    frame.loc[np.asarray(list(support_hat), dtype=int), "selected_for_support_eval"] = 1
    frame["support_eval_mode"] = support_mode
    return frame


def _selected_valid_metric(model: object) -> float | None:
    trace = getattr(model, "selection_trace_", None)
    if not isinstance(trace, pd.DataFrame) or trace.empty or "valid_mse" not in trace.columns:
        return None
    if getattr(model, "selected_checkpoint_", None) is not None and "checkpoint" in trace.columns:
        sub = trace.loc[trace["checkpoint"].astype(int) == int(model.selected_checkpoint_)]
        if not sub.empty:
            return float(sub["valid_mse"].iloc[0])
    if getattr(model, "selected_step_", None) is not None and "checkpoint" in trace.columns:
        sub = trace.loc[trace["checkpoint"].astype(int) == int(model.selected_step_)]
        if not sub.empty:
            return float(sub["valid_mse"].iloc[0])
    if getattr(model, "selected_alpha_", None) is not None and "alpha" in trace.columns:
        sub = trace.loc[np.isclose(trace["alpha"].astype(float), float(model.selected_alpha_))]
        if not sub.empty:
            return float(sub["valid_mse"].iloc[0])
    return None


def _trial_row(*, design: str, rep: int, family_name: str, model_name: str, dataset, model: object, support_hat: np.ndarray, support_mode: str) -> Dict[str, object]:
    test_pred = np.asarray(model.predict(dataset.test.X), dtype=float)
    row: Dict[str, object] = {
        "design": design,
        "rep": rep,
        "seed": int(dataset.config["seed"]),
        "family_name": family_name,
        "model_name": model_name,
        "p": int(dataset.train.X.shape[1]),
        "s": int(dataset.support_true.shape[0]),
        "support_eval_mode": support_mode,
        "support_hat_json": json.dumps([int(x) for x in np.asarray(support_hat, dtype=int).tolist()]),
    }
    valid_mse_selected = _selected_valid_metric(model)
    if valid_mse_selected is not None:
        row["valid_mse_selected"] = float(valid_mse_selected)
    row.update({f"test_{k}": v for k, v in regression_metrics(dataset.test.y, test_pred).items()})
    row.update({f"support_{k}": v for k, v in support_recovery_metrics(dataset.support_true, support_hat, p=dataset.train.X.shape[1]).items()})

    if getattr(model, "selection_trace_", None) is not None:
        row["selection_trace_rows"] = int(len(model.selection_trace_))
    if getattr(model, "selected_step_", None) is not None:
        row["selected_step"] = int(model.selected_step_)
    if getattr(model, "selected_checkpoint_", None) is not None:
        row["selected_checkpoint"] = int(model.selected_checkpoint_)
    if getattr(model, "selected_alpha_", None) is not None:
        row["selected_alpha"] = float(model.selected_alpha_)
    row["selected_support_size"] = int(np.asarray(support_hat, dtype=int).shape[0])
    return row


def _run_model_trial(task: Dict[str, object]) -> Dict[str, object]:
    design = str(task["design"])
    rep = int(task["rep"])
    seed = int(task["base_seed"]) + rep
    base_model_name = str(task["base_model_name"])
    family_name = str(task.get("family_name", base_model_name))
    model_name = str(task.get("model_name", base_model_name))
    dataset = generate_sparse_regression_dataset(
        n_train=int(task["n_train"]),
        n_valid=int(task["n_valid"]),
        n_test=int(task["n_test"]),
        p=int(task["p"]),
        s=int(task["s"]),
        design=design,
        rho=float(task["rho"]),
        block_size=int(task["block_size"]),
        beta_scale=float(task["beta_scale"]),
        beta_pattern=str(task["beta_pattern"]),
        support_strategy=str(task["support_strategy"]),
        snr=float(task["snr"]),
        seed=seed,
    )
    dataset_summary = summarize_sparse_regression_dataset(dataset)
    dataset_summary.update({"design": design, "rep": rep, "seed": seed})

    model_kwargs = {"random_state": seed}
    if base_model_name == "ctb_sparse":
        model_kwargs.update(
            max_steps=int(task["ctb_max_steps"]),
            n_inner_bootstraps=int(task["ctb_inner_bootstraps"]),
            eta=float(task["ctb_eta"]),
            residual_weight_power=float(task["ctb_residual_weight_power"]),
            residual_weight_eps=float(task["ctb_residual_weight_eps"]),
            consensus_frequency_power=float(task["ctb_consensus_frequency_power"]),
            consensus_sign_power=float(task["ctb_consensus_sign_power"]),
            instability_lambda=float(task["ctb_instability_lambda"]),
            instability_power=float(task["ctb_instability_power"]),
            min_consensus_frequency=float(task["ctb_min_consensus_frequency"]),
            min_sign_consistency=float(task["ctb_min_sign_consistency"]),
            support_frequency_threshold=float(task["ctb_support_frequency_threshold"]),
        )
    if base_model_name == "ctb_tree":
        model_kwargs.update(
            n_estimators=int(task["ctb_max_steps"]),
            n_inner_bootstraps=int(task["ctb_inner_bootstraps"]),
            eta=float(task["ctb_eta"]),
            max_depth=int(task["ctb_tree_max_depth"]),
            min_samples_leaf=int(task["ctb_tree_min_samples_leaf"]),
            update_target_mode=str(task["ctb_target_mode"]),
            transport_curvature_eps=float(task["ctb_curvature_eps"]),
            instability_penalty=0.0,
            weight_power=float(task["ctb_residual_weight_power"]),
            weight_eps=float(task["ctb_residual_weight_eps"]),
        )
    model = build_experiment4_model(model_name=base_model_name, **model_kwargs)
    model.fit(dataset.train, dataset.valid)
    support_hat, support_mode = _resolve_support_hat(family_name, model, dataset, task["xgb_support_k"])

    trace = getattr(model, "selection_trace_", None)
    if trace is not None:
        trace = trace.copy()
        trace.insert(0, "model_name", model_name)
        trace.insert(0, "family_name", family_name)
        trace.insert(0, "rep", rep)
        trace.insert(0, "design", design)
        _save_table(trace, Path(task["traces_dir"]) / f"{design}__rep{rep:02d}__{model_name}.csv")

    if bool(task["save_feature_tables"]):
        feature_frame = _feature_frame_for_model(family_name, model, dataset, support_hat, support_mode)
        feature_frame.insert(0, "model_name", model_name)
        feature_frame.insert(0, "family_name", family_name)
        feature_frame.insert(0, "rep", rep)
        feature_frame.insert(0, "design", design)
        _save_table(feature_frame, Path(task["features_dir"]) / f"{design}__rep{rep:02d}__{model_name}.csv")

    return {
        "trial_row": _trial_row(
            design=design,
            rep=rep,
            family_name=family_name,
            model_name=model_name,
            dataset=dataset,
            model=model,
            support_hat=support_hat,
            support_mode=support_mode,
        ),
        "dataset_summary": dataset_summary,
    }


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    traces_dir = outdir / "selection_traces"
    features_dir = outdir / "feature_tables"

    config = {
        "designs": args.designs,
        "models": args.models,
        "num_seeds": args.num_seeds,
        "base_seed": args.base_seed,
        "n_train": args.n_train,
        "n_valid": args.n_valid,
        "n_test": args.n_test,
        "p": args.p,
        "s": args.s,
        "rho": args.rho,
        "block_size": args.block_size,
        "beta_scale": args.beta_scale,
        "beta_pattern": args.beta_pattern,
        "support_strategy": args.support_strategy,
        "snr": args.snr,
        "xgb_support_k": args.xgb_support_k,
        "ctb_max_steps": args.ctb_max_steps,
        "ctb_inner_bootstraps": args.ctb_inner_bootstraps,
        "ctb_eta": args.ctb_eta,
        "ctb_residual_weight_power": args.ctb_residual_weight_power,
        "ctb_residual_weight_eps": args.ctb_residual_weight_eps,
        "ctb_consensus_frequency_power": args.ctb_consensus_frequency_power,
        "ctb_consensus_sign_power": args.ctb_consensus_sign_power,
        "ctb_instability_lambda": args.ctb_instability_lambda,
        "ctb_instability_power": args.ctb_instability_power,
        "ctb_min_consensus_frequency": args.ctb_min_consensus_frequency,
        "ctb_min_sign_consistency": args.ctb_min_sign_consistency,
        "ctb_support_frequency_threshold": args.ctb_support_frequency_threshold,
        "ctb_tree_max_depths": [int(x) for x in args.ctb_tree_max_depths],
        "ctb_tree_min_samples_leaf": int(args.ctb_tree_min_samples_leaf),
        "ctb_target_modes": [str(x) for x in args.ctb_target_modes],
        "ctb_curvature_eps": [float(x) for x in args.ctb_curvature_eps],
        "n_jobs": args.n_jobs,
        "save_feature_tables": bool(args.save_feature_tables),
    }
    _save_json(config, outdir / "run_config.json")

    trial_rows: List[Dict[str, object]] = []
    dataset_row_map: Dict[tuple[str, int], Dict[str, object]] = {}

    tasks: List[Dict[str, object]] = []
    for design in args.designs:
        for rep in range(args.num_seeds):
            for requested_model_name in args.models:
                if str(requested_model_name) == "ctb_tree":
                    for depth in args.ctb_tree_max_depths:
                        for target_mode in args.ctb_target_modes:
                            for curvature_eps in args.ctb_curvature_eps:
                                candidate_name = ctb_tree_model_name(
                                    depth=int(depth),
                                    update_target_mode=str(target_mode),
                                    transport_curvature_eps=float(curvature_eps),
                                    include_task_suffix=False,
                                ).replace("ctb_depth", "ctb_tree_depth", 1)
                                tasks.append(
                                    {
                                        "design": design,
                                        "rep": rep,
                                        "base_model_name": "ctb_tree",
                                        "family_name": "ctb_tree",
                                        "model_name": candidate_name,
                                        "ctb_tree_max_depth": int(depth),
                                        "ctb_tree_min_samples_leaf": int(args.ctb_tree_min_samples_leaf),
                                        "ctb_target_mode": str(target_mode),
                                        "ctb_curvature_eps": float(curvature_eps),
                                        "base_seed": int(args.base_seed),
                                        "n_train": int(args.n_train),
                                        "n_valid": int(args.n_valid),
                                        "n_test": int(args.n_test),
                                        "p": int(args.p),
                                        "s": int(args.s),
                                        "rho": float(args.rho),
                                        "block_size": int(args.block_size),
                                        "beta_scale": float(args.beta_scale),
                                        "beta_pattern": args.beta_pattern,
                                        "support_strategy": args.support_strategy,
                                        "snr": float(args.snr),
                                        "xgb_support_k": args.xgb_support_k,
                                        "ctb_max_steps": int(args.ctb_max_steps),
                                        "ctb_inner_bootstraps": int(args.ctb_inner_bootstraps),
                                        "ctb_eta": float(args.ctb_eta),
                                        "ctb_residual_weight_power": float(args.ctb_residual_weight_power),
                                        "ctb_residual_weight_eps": float(args.ctb_residual_weight_eps),
                                        "ctb_consensus_frequency_power": float(args.ctb_consensus_frequency_power),
                                        "ctb_consensus_sign_power": float(args.ctb_consensus_sign_power),
                                        "ctb_instability_lambda": float(args.ctb_instability_lambda),
                                        "ctb_instability_power": float(args.ctb_instability_power),
                                        "ctb_min_consensus_frequency": float(args.ctb_min_consensus_frequency),
                                        "ctb_min_sign_consistency": float(args.ctb_min_sign_consistency),
                                        "ctb_support_frequency_threshold": float(args.ctb_support_frequency_threshold),
                                        "save_feature_tables": bool(args.save_feature_tables),
                                        "traces_dir": str(traces_dir),
                                        "features_dir": str(features_dir),
                                    }
                                )
                    continue
                tasks.append(
                    {
                        "design": design,
                        "rep": rep,
                        "base_model_name": str(requested_model_name),
                        "family_name": str(requested_model_name),
                        "model_name": str(requested_model_name),
                        "base_seed": int(args.base_seed),
                        "n_train": int(args.n_train),
                        "n_valid": int(args.n_valid),
                        "n_test": int(args.n_test),
                        "p": int(args.p),
                        "s": int(args.s),
                        "rho": float(args.rho),
                        "block_size": int(args.block_size),
                        "beta_scale": float(args.beta_scale),
                        "beta_pattern": args.beta_pattern,
                        "support_strategy": args.support_strategy,
                        "snr": float(args.snr),
                        "xgb_support_k": args.xgb_support_k,
                        "ctb_max_steps": int(args.ctb_max_steps),
                        "ctb_inner_bootstraps": int(args.ctb_inner_bootstraps),
                        "ctb_eta": float(args.ctb_eta),
                        "ctb_residual_weight_power": float(args.ctb_residual_weight_power),
                        "ctb_residual_weight_eps": float(args.ctb_residual_weight_eps),
                        "ctb_consensus_frequency_power": float(args.ctb_consensus_frequency_power),
                        "ctb_consensus_sign_power": float(args.ctb_consensus_sign_power),
                        "ctb_instability_lambda": float(args.ctb_instability_lambda),
                        "ctb_instability_power": float(args.ctb_instability_power),
                        "ctb_min_consensus_frequency": float(args.ctb_min_consensus_frequency),
                        "ctb_min_sign_consistency": float(args.ctb_min_sign_consistency),
                        "ctb_support_frequency_threshold": float(args.ctb_support_frequency_threshold),
                        "save_feature_tables": bool(args.save_feature_tables),
                        "traces_dir": str(traces_dir),
                        "features_dir": str(features_dir),
                    }
                )

    n_jobs = resolve_n_jobs(args.n_jobs)
    with progress_bar(total=len(tasks), desc="Experiment 4 benchmark", unit="model") as pbar:
        if n_jobs <= 1:
            results = []
            for task in tasks:
                results.append(_run_model_trial(task))
                pbar.update(1)
        else:
            results = []
            with make_process_pool(n_jobs) as executor:
                futures = [executor.submit(_run_model_trial, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

    for result in results:
        trial_rows.append(result["trial_row"])
        dataset_summary = result["dataset_summary"]
        dataset_key = (str(dataset_summary["design"]), int(dataset_summary["rep"]))
        dataset_row_map.setdefault(dataset_key, dataset_summary)

    trial_df = pd.DataFrame(trial_rows)
    dataset_df = pd.DataFrame(dataset_row_map.values())
    _save_table(trial_df, outdir / "trial_metrics.csv")
    if not trial_df.empty and "family_name" in trial_df.columns and "valid_mse_selected" in trial_df.columns:
        family_selected_idx = trial_df.groupby(["design", "rep", "family_name"], dropna=False)["valid_mse_selected"].idxmin().dropna().astype(int).tolist()
        family_selected_df = trial_df.loc[family_selected_idx].copy()
        _save_table(family_selected_df, outdir / "trial_metrics_family_selected.csv")
    _save_table(dataset_df, outdir / "dataset_summaries.csv")

    quick_summary = {
        "n_trials": float(len(trial_df)),
        "designs": sorted(trial_df["design"].astype(str).unique().tolist()) if not trial_df.empty else [],
        "families": sorted(trial_df["family_name"].astype(str).unique().tolist()) if (not trial_df.empty and "family_name" in trial_df.columns) else [],
        "models": sorted(trial_df["model_name"].astype(str).unique().tolist()) if not trial_df.empty else [],
    }
    _save_json(quick_summary, outdir / "artifact_summary.json")
    print(json.dumps(quick_summary, indent=2))
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
