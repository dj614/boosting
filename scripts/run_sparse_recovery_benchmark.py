#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from progress_utils import progress_bar
from sim.sparse_recovery_data import generate_sparse_regression_dataset, summarize_sparse_regression_dataset
from sim.sparse_recovery_eval import make_feature_support_frame, regression_metrics, support_recovery_metrics
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
        choices=["l2boost", "bagged_componentwise", "lasso", "xgb_tree"],
        default=["l2boost", "bagged_componentwise", "lasso", "xgb_tree"],
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
    parser.add_argument("--save-feature-tables", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment4_sparse_recovery"))
    return parser


def _save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _resolve_support_hat(model_name: str, model: object, dataset, xgb_support_k: int | None) -> tuple[np.ndarray, str]:
    if model_name == "xgb_tree":
        k = int(xgb_support_k or dataset.support_true.shape[0])
        support_hat = np.asarray(model.topk_support(k), dtype=int)
        return support_hat, f"topk_importance@{k}"
    support_hat = np.asarray(getattr(model, "selected_support_", np.array([], dtype=int)), dtype=int)
    return support_hat, "native"


def _feature_frame_for_model(model_name: str, model: object, dataset, support_hat: Sequence[int], support_mode: str) -> pd.DataFrame:
    p = dataset.train.X.shape[1]
    feature_names = dataset.feature_names
    extra_columns = {
        "corr_to_signal": dataset.train.meta["corr_to_signal"],
        "true_beta": dataset.beta_true,
    }

    if model_name == "bagged_componentwise" and getattr(model, "selection_frequency_", None) is not None:
        frame = make_feature_support_frame(
            selection_frequency=np.asarray(model.selection_frequency_, dtype=float),
            support_true=dataset.support_true,
            feature_names=feature_names,
            extra_columns=extra_columns,
        )
    elif model_name == "xgb_tree" and getattr(model, "feature_importances_", None) is not None:
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


def _trial_row(*, design: str, rep: int, model_name: str, dataset, model: object, support_hat: np.ndarray, support_mode: str) -> Dict[str, object]:
    test_pred = np.asarray(model.predict(dataset.test.X), dtype=float)
    row: Dict[str, object] = {
        "design": design,
        "rep": rep,
        "seed": int(dataset.config["seed"]),
        "model_name": model_name,
        "p": int(dataset.train.X.shape[1]),
        "s": int(dataset.support_true.shape[0]),
        "support_eval_mode": support_mode,
        "support_hat_json": json.dumps([int(x) for x in np.asarray(support_hat, dtype=int).tolist()]),
    }
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
        "save_feature_tables": bool(args.save_feature_tables),
    }
    _save_json(config, outdir / "run_config.json")

    trial_rows: List[Dict[str, object]] = []
    dataset_rows: List[Dict[str, object]] = []

    total_model_fits = len(args.designs) * args.num_seeds * len(args.models)

    with progress_bar(total=total_model_fits, desc="Experiment 4 benchmark", unit="model") as pbar:
        for design in args.designs:
            for rep in range(args.num_seeds):
                seed = args.base_seed + rep
                dataset = generate_sparse_regression_dataset(
                    n_train=args.n_train,
                    n_valid=args.n_valid,
                    n_test=args.n_test,
                    p=args.p,
                    s=args.s,
                    design=design,
                    rho=args.rho,
                    block_size=args.block_size,
                    beta_scale=args.beta_scale,
                    beta_pattern=args.beta_pattern,
                    support_strategy=args.support_strategy,
                    snr=args.snr,
                    seed=seed,
                )
                summary = summarize_sparse_regression_dataset(dataset)
                summary.update({"design": design, "rep": rep, "seed": seed})
                dataset_rows.append(summary)

                for model_name in args.models:
                    pbar.set_postfix(design=design, rep=rep, model=model_name)
                    model = build_experiment4_model(model_name=model_name, random_state=seed)
                    model.fit(dataset.train, dataset.valid)
                    support_hat, support_mode = _resolve_support_hat(model_name, model, dataset, args.xgb_support_k)
                    trial_rows.append(
                        _trial_row(
                            design=design,
                            rep=rep,
                            model_name=model_name,
                            dataset=dataset,
                            model=model,
                            support_hat=support_hat,
                            support_mode=support_mode,
                        )
                    )

                    trace = getattr(model, "selection_trace_", None)
                    if trace is not None:
                        trace = trace.copy()
                        trace.insert(0, "model_name", model_name)
                        trace.insert(0, "rep", rep)
                        trace.insert(0, "design", design)
                        _save_table(trace, traces_dir / f"{design}__rep{rep:02d}__{model_name}.csv")

                    if args.save_feature_tables:
                        feature_frame = _feature_frame_for_model(model_name, model, dataset, support_hat, support_mode)
                        feature_frame.insert(0, "model_name", model_name)
                        feature_frame.insert(0, "rep", rep)
                        feature_frame.insert(0, "design", design)
                        _save_table(feature_frame, features_dir / f"{design}__rep{rep:02d}__{model_name}.csv")
                    pbar.update(1)

    trial_df = pd.DataFrame(trial_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    _save_table(trial_df, outdir / "trial_metrics.csv")
    _save_table(dataset_df, outdir / "dataset_summaries.csv")

    quick_summary = {
        "n_trials": float(len(trial_df)),
        "designs": sorted(trial_df["design"].astype(str).unique().tolist()) if not trial_df.empty else [],
        "models": sorted(trial_df["model_name"].astype(str).unique().tolist()) if not trial_df.empty else [],
    }
    _save_json(quick_summary, outdir / "artifact_summary.json")
    print(json.dumps(quick_summary, indent=2))
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
