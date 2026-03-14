#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CLASSIFICATION_METRICS = [
    "test_accuracy",
    "test_balanced_accuracy",
    "test_log_loss",
    "test_roc_auc",
    "test_calibration_error",
]
DEFAULT_REGRESSION_METRICS = [
    "test_rmse",
    "test_mae",
    "test_r2",
]
DEFAULT_FAMILY_ORDER = ["bagging", "rf", "gbdt", "xgb", "ctb_dt", "ctb_xgbtree", "ctb"]
PRIMARY_METRIC = {
    "classification": "log_loss",
    "regression": "rmse",
}


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze open tabular benchmark outputs and render PNG comparisons.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory produced by scripts/run_open_tabular_benchmark.py",
    )
    parser.add_argument("--outdir", type=Path, default=None, help="Defaults to <input-dir>/analysis")
    parser.add_argument(
        "--task-types",
        nargs="*",
        default=None,
        help="Optional subset of task types to analyze, e.g. classification regression",
    )
    parser.add_argument(
        "--classification-metrics",
        nargs="*",
        default=DEFAULT_CLASSIFICATION_METRICS,
        help="Classification test metrics to summarize and plot.",
    )
    parser.add_argument(
        "--regression-metrics",
        nargs="*",
        default=DEFAULT_REGRESSION_METRICS,
        help="Regression test metrics to summarize and plot.",
    )
    parser.add_argument(
        "--family-order",
        nargs="*",
        default=DEFAULT_FAMILY_ORDER,
        help="Preferred ordering for model families in tables/plots.",
    )
    return parser


def _metric_preference(metric_name: str) -> str:
    metric = str(metric_name).lower()
    if any(token in metric for token in ["log_loss", "brier", "calibration_error", "rmse", "mae"]):
        return "lower"
    return "higher"


def _ordered_unique(values: Iterable[str], preferred_order: Sequence[str]) -> List[str]:
    seen = {str(v) for v in values}
    ordered = [item for item in preferred_order if item in seen]
    ordered.extend(sorted(seen.difference(ordered)))
    return ordered


def _task_metrics(task_type: str, args: argparse.Namespace, available_columns: Sequence[str]) -> List[str]:
    if task_type == "classification":
        requested = list(args.classification_metrics)
    elif task_type == "regression":
        requested = list(args.regression_metrics)
    else:
        requested = []
    return [metric for metric in requested if metric in available_columns]


def _safe_float(value: object) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return float(value)


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _format_annotation(value: float) -> str:
    if not np.isfinite(value):
        return ""
    abs_value = abs(float(value))
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _annotated_heatmap(frame: pd.DataFrame, *, title: str, path: Path, cmap: str = "viridis") -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    plot_frame = plot_frame.astype(float)
    values = plot_frame.to_numpy(dtype=float)

    fig_width = max(6.0, 1.2 * max(1, plot_frame.shape[1]))
    fig_height = max(4.0, 0.55 * max(1, plot_frame.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(plot_frame.shape[1]))
    ax.set_xticklabels([str(x) for x in plot_frame.columns], rotation=30, ha="right")
    ax.set_yticks(range(plot_frame.shape[0]))
    ax.set_yticklabels([str(x) for x in plot_frame.index])
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("value", rotation=270, labelpad=12)

    finite_values = values[np.isfinite(values)]
    threshold = float(np.nanmedian(finite_values)) if finite_values.size > 0 else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value >= threshold else "black"
            ax.text(j, i, _format_annotation(float(value)), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _grouped_bar_plot(
    frame: pd.DataFrame,
    *,
    x_col: str,
    series_col: str,
    value_col: str,
    title: str,
    ylabel: str,
    path: Path,
    hline_zero: bool = False,
) -> None:
    if frame.empty:
        return
    x_values = [str(x) for x in frame[x_col].dropna().astype(str).unique().tolist()]
    series_values = [str(x) for x in frame[series_col].dropna().astype(str).unique().tolist()]
    if not x_values or not series_values:
        return

    pivot = frame.pivot(index=x_col, columns=series_col, values=value_col).reindex(index=x_values, columns=series_values)
    n_series = len(series_values)
    width = min(0.8 / max(1, n_series), 0.22)
    x_pos = np.arange(len(x_values), dtype=float)

    fig_width = max(7.0, 0.9 * len(x_values) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    offsets = (np.arange(n_series, dtype=float) - 0.5 * (n_series - 1)) * width
    for idx, series_name in enumerate(series_values):
        series_y = pivot[series_name].to_numpy(dtype=float)
        ax.bar(x_pos + offsets[idx], series_y, width=width, label=series_name)

    if hline_zero:
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_values, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=series_col, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_summary_frame(input_dir: Path) -> pd.DataFrame:
    summary_path = input_dir / "summary_test_metrics.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    frame = pd.read_csv(summary_path)
    if frame.empty:
        raise ValueError("summary_test_metrics.csv is empty")
    required = {"task_type", "dataset_name", "repeat_id", "family"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"summary_test_metrics.csv missing required columns: {sorted(missing)}")
    return frame


def _aggregate_dataset_family_metrics(frame: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["task_type", "dataset_name", "family"]
    for group_key, sub_df in frame.groupby(group_cols, dropna=False):
        row: Dict[str, object] = {
            "task_type": str(group_key[0]),
            "dataset_name": str(group_key[1]),
            "family": str(group_key[2]),
            "n_repeats": int(sub_df["repeat_id"].nunique()),
        }
        for metric in metric_cols:
            values = pd.to_numeric(sub_df[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        if "selected_checkpoint" in sub_df.columns:
            checkpoints = pd.to_numeric(sub_df["selected_checkpoint"], errors="coerce")
            row["selected_checkpoint_mean"] = float(checkpoints.mean())
            row["selected_checkpoint_std"] = float(checkpoints.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _family_rank_summary(agg_df: pd.DataFrame, metric_cols: Sequence[str], family_order: Sequence[str]) -> pd.DataFrame:
    rank_rows: List[Dict[str, object]] = []
    for task_type, task_df in agg_df.groupby("task_type", dropna=False):
        for metric in metric_cols:
            mean_col = f"{metric}_mean"
            if mean_col not in task_df.columns:
                continue
            metric_ranks: List[pd.DataFrame] = []
            ascending = _metric_preference(metric) == "lower"
            for dataset_name, dataset_df in task_df.groupby("dataset_name", dropna=False):
                ranked = dataset_df[["family", mean_col]].copy()
                ranked["rank"] = ranked[mean_col].rank(method="average", ascending=ascending)
                ranked["dataset_name"] = str(dataset_name)
                metric_ranks.append(ranked[["dataset_name", "family", "rank"]])
            if not metric_ranks:
                continue
            rank_df = pd.concat(metric_ranks, ignore_index=True)
            summary = rank_df.groupby("family", dropna=False)["rank"].mean().reset_index()
            summary["task_type"] = str(task_type)
            summary["metric"] = str(metric)
            summary.rename(columns={"rank": "average_rank"}, inplace=True)
            rank_rows.append(summary)
    if not rank_rows:
        return pd.DataFrame(columns=["task_type", "metric", "family", "average_rank"])
    rank_summary = pd.concat(rank_rows, ignore_index=True)
    rank_summary["family"] = pd.Categorical(
        rank_summary["family"].astype(str),
        categories=_ordered_unique(rank_summary["family"].astype(str), family_order),
        ordered=True,
    )
    return rank_summary.sort_values(["task_type", "metric", "family"]).reset_index(drop=True)


def _ctb_advantage_summary(agg_df: pd.DataFrame, metric_cols: Sequence[str], family_order: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (task_type, dataset_name), dataset_df in agg_df.groupby(["task_type", "dataset_name"], dropna=False):
        family_values = {str(f): dataset_df.loc[dataset_df["family"] == f].iloc[0] for f in dataset_df["family"].astype(str).unique()}
        if "ctb" not in family_values:
            continue
        ctb_row = family_values["ctb"]
        baseline_families = [family for family in _ordered_unique(family_values.keys(), family_order) if family != "ctb"]
        for baseline_family in baseline_families:
            baseline_row = family_values[baseline_family]
            for metric in metric_cols:
                mean_col = f"{metric}_mean"
                if mean_col not in dataset_df.columns:
                    continue
                ctb_value = _safe_float(ctb_row.get(mean_col))
                baseline_value = _safe_float(baseline_row.get(mean_col))
                if _metric_preference(metric) == "lower":
                    advantage = baseline_value - ctb_value
                else:
                    advantage = ctb_value - baseline_value
                rows.append(
                    {
                        "task_type": str(task_type),
                        "dataset_name": str(dataset_name),
                        "baseline_family": str(baseline_family),
                        "metric": str(metric),
                        "ctb_value": ctb_value,
                        "baseline_value": baseline_value,
                        "ctb_advantage": float(advantage),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["task_type", "dataset_name", "baseline_family", "metric", "ctb_value", "baseline_value", "ctb_advantage"]
        )
    result = pd.DataFrame(rows)
    result["baseline_family"] = pd.Categorical(
        result["baseline_family"].astype(str),
        categories=[family for family in _ordered_unique(result["baseline_family"].astype(str), family_order) if family != "ctb"],
        ordered=True,
    )
    return result.sort_values(["task_type", "metric", "baseline_family", "dataset_name"]).reset_index(drop=True)


def _best_family_summary(agg_df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (task_type, dataset_name), dataset_df in agg_df.groupby(["task_type", "dataset_name"], dropna=False):
        for metric in metric_cols:
            mean_col = f"{metric}_mean"
            if mean_col not in dataset_df.columns:
                continue
            ascending = _metric_preference(metric) == "lower"
            ranked = dataset_df.sort_values(mean_col, ascending=ascending).reset_index(drop=True)
            best = ranked.iloc[0]
            rows.append(
                {
                    "task_type": str(task_type),
                    "dataset_name": str(dataset_name),
                    "metric": str(metric),
                    "best_family": str(best["family"]),
                    "best_value": float(best[mean_col]),
                }
            )
    return pd.DataFrame(rows).sort_values(["task_type", "dataset_name", "metric"]).reset_index(drop=True)


def _generalization_gap_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for task_type, metric_suffix in PRIMARY_METRIC.items():
        valid_col = f"valid_{metric_suffix}"
        test_col = f"test_{metric_suffix}"
        if valid_col not in frame.columns or test_col not in frame.columns:
            continue
        task_df = frame.loc[frame["task_type"].astype(str) == task_type].copy()
        if task_df.empty:
            continue
        grouped = task_df.groupby(["task_type", "dataset_name", "family"], dropna=False)
        for (group_task_type, dataset_name, family), sub_df in grouped:
            valid_vals = pd.to_numeric(sub_df[valid_col], errors="coerce")
            test_vals = pd.to_numeric(sub_df[test_col], errors="coerce")
            rows.append(
                {
                    "task_type": str(group_task_type),
                    "dataset_name": str(dataset_name),
                    "family": str(family),
                    "primary_metric": str(metric_suffix),
                    "mean_generalization_gap": float((test_vals - valid_vals).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["task_type", "dataset_name", "family"]).reset_index(drop=True)


def _plot_task_heatmaps(
    *,
    task_type: str,
    metrics: Sequence[str],
    agg_df: pd.DataFrame,
    family_order: Sequence[str],
    plot_dir: Path,
) -> None:
    task_df = agg_df.loc[agg_df["task_type"].astype(str) == task_type].copy()
    if task_df.empty:
        return
    families = _ordered_unique(task_df["family"].astype(str), family_order)
    datasets = sorted(task_df["dataset_name"].astype(str).unique().tolist())
    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in task_df.columns:
            continue
        pivot = task_df.pivot(index="dataset_name", columns="family", values=mean_col).reindex(index=datasets, columns=families)
        _annotated_heatmap(
            pivot,
            title=f"{task_type}: mean {metric}",
            path=plot_dir / f"{task_type}_{metric}_heatmap.png",
            cmap="viridis_r" if _metric_preference(metric) == "lower" else "viridis",
        )


def _plot_rank_heatmap(task_type: str, rank_df: pd.DataFrame, family_order: Sequence[str], plot_dir: Path) -> None:
    task_rank_df = rank_df.loc[rank_df["task_type"].astype(str) == task_type].copy()
    if task_rank_df.empty:
        return
    families = _ordered_unique(task_rank_df["family"].astype(str), family_order)
    metrics = task_rank_df["metric"].astype(str).drop_duplicates().tolist()
    pivot = task_rank_df.pivot(index="family", columns="metric", values="average_rank").reindex(index=families, columns=metrics)
    _annotated_heatmap(
        pivot,
        title=f"{task_type}: average family rank across datasets (lower is better)",
        path=plot_dir / f"{task_type}_average_family_ranks.png",
        cmap="viridis_r",
    )


def _plot_ctb_advantages(task_type: str, ctb_df: pd.DataFrame, plot_dir: Path) -> None:
    task_ctb_df = ctb_df.loc[ctb_df["task_type"].astype(str) == task_type].copy()
    if task_ctb_df.empty:
        return
    for metric in task_ctb_df["metric"].astype(str).drop_duplicates().tolist():
        metric_df = task_ctb_df.loc[task_ctb_df["metric"].astype(str) == metric].copy()
        ylabel = "ctb advantage (+ means ctb better)"
        _grouped_bar_plot(
            metric_df,
            x_col="dataset_name",
            series_col="baseline_family",
            value_col="ctb_advantage",
            title=f"{task_type}: ctb vs baselines on {metric}",
            ylabel=ylabel,
            path=plot_dir / f"{task_type}_ctb_vs_baselines_{metric}.png",
            hline_zero=True,
        )


def _plot_generalization_gap(task_type: str, gap_df: pd.DataFrame, family_order: Sequence[str], plot_dir: Path) -> None:
    task_gap_df = gap_df.loc[gap_df["task_type"].astype(str) == task_type].copy()
    if task_gap_df.empty:
        return
    families = _ordered_unique(task_gap_df["family"].astype(str), family_order)
    datasets = sorted(task_gap_df["dataset_name"].astype(str).unique().tolist())
    pivot = task_gap_df.pivot(index="dataset_name", columns="family", values="mean_generalization_gap").reindex(index=datasets, columns=families)
    primary_metric = str(task_gap_df["primary_metric"].iloc[0]) if not task_gap_df.empty else "metric"
    _annotated_heatmap(
        pivot,
        title=f"{task_type}: test-valid gap for primary metric ({primary_metric})",
        path=plot_dir / f"{task_type}_primary_metric_generalization_gap.png",
        cmap="coolwarm",
    )


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    outdir: Path = args.outdir or (input_dir / "analysis")
    plot_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _load_summary_frame(input_dir)
    available_task_types = sorted(summary_df["task_type"].dropna().astype(str).unique().tolist())
    if args.task_types:
        task_types = [task_type for task_type in args.task_types if task_type in available_task_types]
    else:
        task_types = available_task_types
    if not task_types:
        raise ValueError(f"No matching task types to analyze. Available task types: {available_task_types}")

    filtered_df = summary_df.loc[summary_df["task_type"].astype(str).isin(task_types)].copy()
    family_order = _ordered_unique(filtered_df["family"].astype(str), args.family_order)

    requested_metric_cols: List[str] = []
    for task_type in task_types:
        requested_metric_cols.extend(_task_metrics(task_type, args, filtered_df.columns))
    primary_valid_test_cols = [
        f"valid_{PRIMARY_METRIC[task_type]}" for task_type in task_types if f"valid_{PRIMARY_METRIC[task_type]}" in filtered_df.columns
    ] + [
        f"test_{PRIMARY_METRIC[task_type]}" for task_type in task_types if f"test_{PRIMARY_METRIC[task_type]}" in filtered_df.columns
    ]
    metric_cols = sorted(set(requested_metric_cols + primary_valid_test_cols))

    agg_df = _aggregate_dataset_family_metrics(filtered_df, metric_cols=metric_cols)
    rank_df = _family_rank_summary(agg_df, metric_cols=requested_metric_cols, family_order=family_order)
    ctb_df = _ctb_advantage_summary(agg_df, metric_cols=requested_metric_cols, family_order=family_order)
    best_df = _best_family_summary(agg_df, metric_cols=requested_metric_cols)
    gap_df = _generalization_gap_summary(filtered_df)

    _save_table(agg_df, outdir / "dataset_family_metric_summary.csv")
    _save_table(rank_df, outdir / "family_average_ranks.csv")
    _save_table(ctb_df, outdir / "ctb_vs_baselines.csv")
    _save_table(best_df, outdir / "best_family_by_dataset_metric.csv")
    _save_table(gap_df, outdir / "primary_metric_generalization_gap.csv")

    summary_payload: Dict[str, object] = {
        "input_dir": str(input_dir),
        "task_types": list(task_types),
        "n_rows": int(len(filtered_df)),
        "families": list(family_order),
        "datasets_by_task": {
            task_type: sorted(filtered_df.loc[filtered_df["task_type"].astype(str) == task_type, "dataset_name"].astype(str).unique().tolist())
            for task_type in task_types
        },
        "metrics_by_task": {
            task_type: _task_metrics(task_type, args, filtered_df.columns)
            for task_type in task_types
        },
    }
    if not ctb_df.empty:
        ctb_win_rows: List[Dict[str, object]] = []
        for (task_type, metric, baseline_family), sub_df in ctb_df.groupby(["task_type", "metric", "baseline_family"], dropna=False, observed=False):
            wins = int((pd.to_numeric(sub_df["ctb_advantage"], errors="coerce") > 0).sum())
            ties = int((pd.to_numeric(sub_df["ctb_advantage"], errors="coerce") == 0).sum())
            total = int(len(sub_df))
            ctb_win_rows.append(
                {
                    "task_type": str(task_type),
                    "metric": str(metric),
                    "baseline_family": str(baseline_family),
                    "ctb_win_count": wins,
                    "tie_count": ties,
                    "n_datasets": total,
                }
            )
        summary_payload["ctb_win_summary"] = ctb_win_rows
    if not best_df.empty:
        summary_payload["best_family_counts"] = (
            best_df.groupby(["task_type", "metric", "best_family"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["task_type", "metric", "count", "best_family"], ascending=[True, True, False, True])
            .to_dict(orient="records")
        )
    _save_json(summary_payload, outdir / "analysis_summary.json")

    for task_type in task_types:
        task_metrics = _task_metrics(task_type, args, filtered_df.columns)
        _plot_task_heatmaps(task_type=task_type, metrics=task_metrics, agg_df=agg_df, family_order=family_order, plot_dir=plot_dir)
        _plot_rank_heatmap(task_type=task_type, rank_df=rank_df, family_order=family_order, plot_dir=plot_dir)
        _plot_ctb_advantages(task_type=task_type, ctb_df=ctb_df, plot_dir=plot_dir)
        _plot_generalization_gap(task_type=task_type, gap_df=gap_df, family_order=family_order, plot_dir=plot_dir)

    print(json.dumps(summary_payload, indent=2))
    print(f"Wrote analysis outputs to: {outdir}")


if __name__ == "__main__":
    main()
