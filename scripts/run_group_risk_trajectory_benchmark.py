#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import as_completed
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parallel_utils import make_process_pool, resolve_n_jobs
from progress_utils import progress_bar

from sim.grouped_classification_data import (  # noqa: E402
    BinaryClassificationDataset,
    BinaryClassificationSplit,
    load_adult_income,
    simulate_grouped_classification,
    summarize_binary_classification_dataset,
    with_margin_based_difficulty_groups,
)
from sim.grouped_classification_eval import (  # noqa: E402
    binary_brier_per_sample,
    binary_log_loss_per_sample,
    evaluate_binary_predictions,
    make_binary_prediction_frame,
    save_prediction_frame,
)
from real_data import load_real_binary_classification_dataset  # noqa: E402
from real_data.schema import (  # noqa: E402
    DEFAULT_REAL_PROCESSED_ROOT,
    DEFAULT_REAL_SPLIT_ROOT,
)
from sim.group_risk_ensemble_models import (  # noqa: E402
    EnsembleModelConfig,
    build_binary_ensemble_wrapper,
    expand_model_grid,
)


DEFAULT_FAMILIES = ["bagging", "rf", "gbdt", "xgb", "ctb"]



def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiment 1 step-2 training loop with staged risk trajectories.")
    parser.add_argument("--dataset", choices=["simulated", "adult", "real"], default="simulated")
    parser.add_argument(
        "--group-definition",
        choices=[
            "auto",
            "sex",
            "age_bucket",
            "education_bucket",
            "sex_age",
            "difficulty",
            "difficulty_group",
            "pclass",
            "sex_pclass",
        ],
        default="sex_age",
        help="Only used for Adult. 'difficulty' means build adult sex_age groups first, then relabel with train-only margins.",
    )
    parser.add_argument("--n-samples", type=int, default=12000, help="Only used for simulated data.")
    parser.add_argument("--n-features", type=int, default=8, help="Only used for simulated data.")
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--real-dataset-name", type=str, default="titanic")
    parser.add_argument("--repeat-id", type=int, default=None)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_REAL_PROCESSED_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_REAL_SPLIT_ROOT)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--max-depths", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--ensemble-sizes", nargs="*", type=int, default=[20, 50, 100, 300])
    parser.add_argument(
        "--trajectory-every",
        type=int,
        default=10,
        help="Add intermediate checkpoints every k learners in addition to --ensemble-sizes. Use 0 to disable.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--ctb-inner-bootstraps", type=int, default=8)
    parser.add_argument("--ctb-eta", type=float, default=1.0)
    parser.add_argument("--ctb-instability-penalty", type=float, default=0.0)
    parser.add_argument("--ctb-weight-power", type=float, default=1.0)
    parser.add_argument("--ctb-weight-eps", type=float, default=1e-8)
    parser.add_argument("--prediction-splits", nargs="*", default=["valid", "test"])
    parser.add_argument("--trajectory-splits", nargs="*", default=["valid", "test"])
    parser.add_argument(
        "--trajectory-sample-count-per-group",
        type=int,
        default=8,
        help="Representative trajectory samples per group and split. Set 0 to skip sample-level trajectory export.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment1_step2"))
    return parser



def _load_dataset(args: argparse.Namespace, seed: int) -> BinaryClassificationDataset:
    if args.dataset == "simulated":
        return simulate_grouped_classification(
            n_samples=args.n_samples,
            n_features=args.n_features,
            valid_size=args.valid_size,
            test_size=args.test_size,
            random_state=seed,
        )
    
    if args.dataset == "real":
        requested_group_definition = args.group_definition
        if requested_group_definition == "difficulty":
            requested_group_definition = "difficulty_group"
        return load_real_binary_classification_dataset(
            dataset_name=args.real_dataset_name,
            repeat_id=int(args.repeat_id) if args.repeat_id is not None else int(seed),
            group_definition=requested_group_definition,
            processed_root=args.processed_root,
            split_root=args.split_root,
            random_state=seed,
        )

    if args.group_definition == "difficulty":
        base = load_adult_income(
            group_definition="sex_age",
            valid_size=args.valid_size,
            test_size=args.test_size,
            random_state=seed,
        )
        return with_margin_based_difficulty_groups(base, random_state=seed)

    return load_adult_income(
        group_definition=args.group_definition,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=seed,
    )



def _trajectory_checkpoints(ensemble_sizes: Sequence[int], every: int) -> List[int]:
    selected = sorted({int(x) for x in ensemble_sizes if int(x) > 0})
    if not selected:
        raise ValueError("ensemble_sizes must be non-empty")
    if every and every > 0:
        for value in range(every, max(selected) + 1, every):
            selected.append(int(value))
    return sorted(set(selected))



def _flatten_metrics(
    evaluation: Dict[str, object],
    *,
    dataset_name: str,
    split: str,
    seed: int,
    model_name: str,
    selected_checkpoint: int,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset_name": dataset_name,
        "split": split,
        "seed": int(seed),
        "model_name": model_name,
        "selected_checkpoint": int(selected_checkpoint),
    }
    row.update({f"overall_{k}": v for k, v in evaluation["overall"].items()})
    row.update({f"core_{k}": v for k, v in evaluation["core_risk"].items()})
    return row



def _split_by_name(dataset: BinaryClassificationDataset, split_name: str) -> BinaryClassificationSplit:
    if split_name == "train":
        return dataset.train
    if split_name == "valid":
        return dataset.valid
    if split_name == "test":
        return dataset.test
    raise ValueError(f"Unsupported split_name={split_name!r}")



def _representative_sample_indices(split: BinaryClassificationSplit, count_per_group: int) -> np.ndarray:
    if count_per_group <= 0:
        return np.zeros(0, dtype=int)

    selected: List[int] = []
    groups = pd.unique(split.group).tolist()
    for group_name in groups:
        group_idx = np.flatnonzero(split.group == group_name)
        if group_idx.size == 0:
            continue
        if split.difficulty_score is not None:
            order = group_idx[np.argsort(np.asarray(split.difficulty_score)[group_idx])]
            take_hard = min(group_idx.size, max(1, count_per_group // 2))
            hard_idx = order[-take_hard:]
            easy_quota = count_per_group - hard_idx.size
            easy_idx = order[: min(group_idx.size, easy_quota)]
            chosen = np.unique(np.concatenate([hard_idx, easy_idx]))
        else:
            chosen = group_idx[: min(group_idx.size, count_per_group)]
        selected.extend(chosen.tolist())
    return np.asarray(sorted(set(selected)), dtype=int)



def _trajectory_core_rows(
    *,
    dataset_name: str,
    split_name: str,
    seed: int,
    model_name: str,
    selected_checkpoint: int,
    split: BinaryClassificationSplit,
    staged_probs: Dict[int, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for checkpoint, y_prob in sorted(staged_probs.items()):
        evaluation = evaluate_binary_predictions(y_true=split.y, y_prob=y_prob, group=split.group)
        row = {
            "dataset_name": dataset_name,
            "split": split_name,
            "seed": int(seed),
            "model_name": model_name,
            "checkpoint": int(checkpoint),
            "selected_checkpoint": int(selected_checkpoint),
            "is_selected": int(int(checkpoint) == int(selected_checkpoint)),
        }
        row.update({f"overall_{k}": v for k, v in evaluation["overall"].items()})
        row.update({f"core_{k}": v for k, v in evaluation["core_risk"].items()})
        rows.append(row)
    return rows



def _trajectory_group_rows(
    *,
    dataset_name: str,
    split_name: str,
    seed: int,
    model_name: str,
    selected_checkpoint: int,
    split: BinaryClassificationSplit,
    staged_probs: Dict[int, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for checkpoint, y_prob in sorted(staged_probs.items()):
        evaluation = evaluate_binary_predictions(y_true=split.y, y_prob=y_prob, group=split.group)
        group_df = evaluation["group_metrics"].copy()
        group_df.insert(0, "is_selected", int(int(checkpoint) == int(selected_checkpoint)))
        group_df.insert(0, "selected_checkpoint", int(selected_checkpoint))
        group_df.insert(0, "checkpoint", int(checkpoint))
        group_df.insert(0, "model_name", model_name)
        group_df.insert(0, "seed", int(seed))
        group_df.insert(0, "split", split_name)
        group_df.insert(0, "dataset_name", dataset_name)
        rows.extend(group_df.to_dict("records"))
    return rows



def _trajectory_sample_rows(
    *,
    dataset_name: str,
    split_name: str,
    seed: int,
    model_name: str,
    selected_checkpoint: int,
    split: BinaryClassificationSplit,
    staged_probs: Dict[int, np.ndarray],
    sample_indices: np.ndarray,
) -> pd.DataFrame:
    if sample_indices.size == 0:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for checkpoint, y_prob in sorted(staged_probs.items()):
        idx = sample_indices
        frame = pd.DataFrame(
            {
                "dataset_name": dataset_name,
                "split": split_name,
                "seed": int(seed),
                "model_name": model_name,
                "checkpoint": int(checkpoint),
                "selected_checkpoint": int(selected_checkpoint),
                "is_selected": int(int(checkpoint) == int(selected_checkpoint)),
                "sample_id": split.sample_id[idx],
                "group": split.group[idx],
                "y_true": split.y[idx],
                "y_prob": np.asarray(y_prob)[idx],
                "log_loss_i": binary_log_loss_per_sample(split.y[idx], np.asarray(y_prob)[idx]),
                "brier_i": binary_brier_per_sample(split.y[idx], np.asarray(y_prob)[idx]),
            }
        )
        if split.difficulty_score is not None:
            frame["difficulty_score"] = np.asarray(split.difficulty_score)[idx]
        if split.bayes_margin is not None:
            frame["bayes_margin"] = np.asarray(split.bayes_margin)[idx]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)



def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _run_model_task(task: Dict[str, object]) -> Dict[str, object]:
    seed = int(task["seed"])
    dataset = _load_dataset(argparse.Namespace(**task["dataset_args"]), seed=seed)
    model_config = EnsembleModelConfig(**task["model_config"])
    model_dir = Path(task["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    wrapper = build_binary_ensemble_wrapper(
        config=model_config,
        selection_checkpoints=task["selection_checkpoints"],
        trajectory_checkpoints=task["trajectory_checkpoints"],
    )
    wrapper.fit(dataset.train, dataset.valid)
    _write_json(model_dir / "model_config.json", model_config.to_dict())
    if wrapper.selection_trace_ is not None:
        wrapper.selection_trace_.to_csv(model_dir / "selection_trace.csv", index=False)

    summary_rows: List[Dict[str, object]] = []
    group_summary_rows: List[Dict[str, object]] = []
    for split_name in task["prediction_splits"]:
        split = _split_by_name(dataset, split_name)
        y_prob = wrapper.predict_proba(split.X)
        evaluation = evaluate_binary_predictions(y_true=split.y, y_prob=y_prob, group=split.group)
        metrics_row = _flatten_metrics(
            evaluation,
            dataset_name=dataset.dataset_name,
            split=split_name,
            seed=seed,
            model_name=model_config.model_name,
            selected_checkpoint=int(wrapper.selected_checkpoint_),
        )
        summary_rows.append(metrics_row)

        group_df = evaluation["group_metrics"].copy()
        group_df.insert(0, "selected_checkpoint", int(wrapper.selected_checkpoint_))
        group_df.insert(0, "model_name", model_config.model_name)
        group_df.insert(0, "seed", int(seed))
        group_df.insert(0, "split", split_name)
        group_df.insert(0, "dataset_name", dataset.dataset_name)
        group_summary_rows.extend(group_df.to_dict("records"))

        _write_json(model_dir / f"metrics_{split_name}.json", {
            "overall": evaluation["overall"],
            "core_risk": evaluation["core_risk"],
            "selected_checkpoint": int(wrapper.selected_checkpoint_),
        })
        group_df.to_csv(model_dir / f"group_metrics_{split_name}.csv", index=False)

        prediction_frame = make_binary_prediction_frame(
            sample_id=split.sample_id,
            dataset_name=dataset.dataset_name,
            split=split_name,
            seed=seed,
            model_name=model_config.model_name,
            group=split.group,
            y_true=split.y,
            y_prob=y_prob,
            metadata=split.metadata,
        )
        save_prediction_frame(prediction_frame, model_dir / f"predictions_{split_name}.csv")

    core_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []
    sample_frames: List[pd.DataFrame] = []
    for split_name in task["trajectory_splits"]:
        split = _split_by_name(dataset, split_name)
        staged_probs = wrapper.trajectory(split.X)
        model_name = model_config.model_name
        core_rows.extend(
            _trajectory_core_rows(
                dataset_name=dataset.dataset_name,
                split_name=split_name,
                seed=seed,
                model_name=model_name,
                selected_checkpoint=int(wrapper.selected_checkpoint_),
                split=split,
                staged_probs=staged_probs,
            )
        )
        group_rows.extend(
            _trajectory_group_rows(
                dataset_name=dataset.dataset_name,
                split_name=split_name,
                seed=seed,
                model_name=model_name,
                selected_checkpoint=int(wrapper.selected_checkpoint_),
                split=split,
                staged_probs=staged_probs,
            )
        )
        sample_idx = _representative_sample_indices(split, int(task["trajectory_sample_count_per_group"]))
        sample_frame = _trajectory_sample_rows(
            dataset_name=dataset.dataset_name,
            split_name=split_name,
            seed=seed,
            model_name=model_name,
            selected_checkpoint=int(wrapper.selected_checkpoint_),
            split=split,
            staged_probs=staged_probs,
            sample_indices=sample_idx,
        )
        if not sample_frame.empty:
            sample_frames.append(sample_frame)

    if core_rows:
        pd.DataFrame(core_rows).to_csv(model_dir / "trajectory_core.csv", index=False)
    if group_rows:
        pd.DataFrame(group_rows).to_csv(model_dir / "trajectory_groups.csv", index=False)
    if sample_frames:
        pd.concat(sample_frames, ignore_index=True).to_csv(model_dir / "trajectory_samples.csv", index=False)

    return {
        "summary_rows": summary_rows,
        "group_summary_rows": group_summary_rows,
    }



def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    families = args.families or list(DEFAULT_FAMILIES)
    selection_checkpoints = sorted({int(x) for x in args.ensemble_sizes if int(x) > 0})
    trajectory_checkpoints = _trajectory_checkpoints(selection_checkpoints, every=args.trajectory_every)

    run_manifest = {
        "dataset": args.dataset,
        "group_definition": args.group_definition,
        "real_dataset_name": args.real_dataset_name,
        "repeat_id": args.repeat_id,
        "processed_root": str(args.processed_root),
        "split_root": str(args.split_root),
        "seed_start": int(args.seed_start),
        "num_seeds": int(args.num_seeds),
        "families": families,
        "max_depths": [int(x) for x in args.max_depths],
        "selection_checkpoints": selection_checkpoints,
        "trajectory_checkpoints": trajectory_checkpoints,
        "prediction_splits": list(args.prediction_splits),
        "trajectory_splits": list(args.trajectory_splits),
        "learning_rate": float(args.learning_rate),
        "min_samples_leaf": int(args.min_samples_leaf),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "ctb_inner_bootstraps": int(args.ctb_inner_bootstraps),
        "ctb_eta": float(args.ctb_eta),
        "ctb_instability_penalty": float(args.ctb_instability_penalty),
        "ctb_weight_power": float(args.ctb_weight_power),
        "ctb_weight_eps": float(args.ctb_weight_eps),
    }
    _write_json(outdir / "run_config.json", run_manifest)

    summary_rows: List[Dict[str, object]] = []
    group_summary_rows: List[Dict[str, object]] = []
    model_grid_template = expand_model_grid(
        families=families,
        max_depths=args.max_depths,
        n_estimators=max(selection_checkpoints),
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        inner_bootstraps=args.ctb_inner_bootstraps,
        eta=args.ctb_eta,
        instability_penalty=args.ctb_instability_penalty,
        weight_power=args.ctb_weight_power,
        weight_eps=args.ctb_weight_eps,
        random_state=args.seed_start,
    )
    total_models = args.num_seeds * len(model_grid_template)

    if args.dataset == "real" and args.repeat_id is None:
        seed_iterable = range(args.seed_start, args.seed_start + args.num_seeds)
    elif args.dataset == "real":
        seed_iterable = [int(args.repeat_id)]
    else:
        seed_iterable = range(args.seed_start, args.seed_start + args.num_seeds)

    dataset_args = {
        "dataset": args.dataset,
        "group_definition": args.group_definition,
        "n_samples": int(args.n_samples),
        "n_features": int(args.n_features),
        "valid_size": float(args.valid_size),
        "test_size": float(args.test_size),
        "real_dataset_name": args.real_dataset_name,
        "repeat_id": args.repeat_id,
        "processed_root": args.processed_root,
        "split_root": args.split_root,
    }
    tasks: List[Dict[str, object]] = []
    for seed in seed_iterable:
        dataset = _load_dataset(args, seed=seed)
        seed_dir = outdir / dataset.dataset_name / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _write_json(seed_dir / "dataset_summary.json", summarize_binary_classification_dataset(dataset))
        model_grid = expand_model_grid(
            families=families,
            max_depths=args.max_depths,
            n_estimators=max(selection_checkpoints),
            learning_rate=args.learning_rate,
            min_samples_leaf=args.min_samples_leaf,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            inner_bootstraps=args.ctb_inner_bootstraps,
            eta=args.ctb_eta,
            instability_penalty=args.ctb_instability_penalty,
            weight_power=args.ctb_weight_power,
            weight_eps=args.ctb_weight_eps,
            random_state=seed,
        )
        for model_config in model_grid:
            tasks.append(
                {
                    "seed": int(seed),
                    "dataset_args": dataset_args,
                    "model_config": model_config.to_dict(),
                    "selection_checkpoints": selection_checkpoints,
                    "trajectory_checkpoints": trajectory_checkpoints,
                    "prediction_splits": list(args.prediction_splits),
                    "trajectory_splits": list(args.trajectory_splits),
                    "trajectory_sample_count_per_group": int(args.trajectory_sample_count_per_group),
                    "model_dir": str(seed_dir / model_config.model_name),
                }
            )

    n_jobs = resolve_n_jobs(args.n_jobs)
    with progress_bar(total=len(tasks), desc="Experiment 2 benchmark", unit="model") as pbar:
        if n_jobs <= 1:
            results = []
            for task in tasks:
                results.append(_run_model_task(task))
                pbar.update(1)
        else:
            results = []
            with make_process_pool(n_jobs) as executor:
                futures = [executor.submit(_run_model_task, task) for task in tasks]
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

    for result in results:
        summary_rows.extend(result["summary_rows"])
        group_summary_rows.extend(result["group_summary_rows"])

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(outdir / "metrics_summary.csv", index=False)
        seed_grouped = summary_df.groupby(["dataset_name", "split", "model_name"], dropna=False)
        mean_df = seed_grouped.mean(numeric_only=True).reset_index()
        se_df = seed_grouped.sem(numeric_only=True).reset_index()
        merged = mean_df.merge(se_df, on=["dataset_name", "split", "model_name"], suffixes=("_mean", "_se"))
        merged.to_csv(outdir / "metrics_summary_aggregated.csv", index=False)
    if group_summary_rows:
        pd.DataFrame(group_summary_rows).to_csv(outdir / "group_metrics_summary.csv", index=False)

    print(f"Wrote step-2 outputs to: {outdir}")


if __name__ == "__main__":
    main()
