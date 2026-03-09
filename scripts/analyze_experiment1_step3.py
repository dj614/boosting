#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sim.experiment1_analysis import (
    build_analysis_summary,
    make_error_variance_scatter,
    make_method_comparison_plot,
    make_pairwise_comparison_table,
    make_slice_heatmaps,
    save_json,
    save_table,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze experiment 1 step-2 outputs and create essay-ready plots.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory produced by scripts/run_experiment1_step2.py",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for analysis artifacts. Defaults to <input-dir>/analysis",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Override primary error metric. Defaults to mse for regression and error_rate for classification.",
    )
    parser.add_argument(
        "--pred-col",
        type=str,
        default=None,
        help="Override pointwise prediction column. Defaults to pred_score for regression, pred_prob for classification.",
    )
    parser.add_argument(
        "--pairwise-baseline",
        type=str,
        default=None,
        help="Optional baseline method for paired difference table. Defaults to the first method alphabetically.",
    )
    parser.add_argument(
        "--skip-heatmaps",
        action="store_true",
        help="Skip 2D slice heatmaps even if pointwise predictions are available.",
    )
    return parser


def _load_run_config(input_dir: Path) -> dict:
    path = input_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_config.json under {input_dir}")
    return json.loads(path.read_text())


def _load_trial_summary(input_dir: Path) -> pd.DataFrame:
    path = input_dir / "trial_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing trial_summary.csv under {input_dir}")
    return pd.read_csv(path)


def _maybe_load_pointwise(input_dir: Path) -> pd.DataFrame | None:
    path = input_dir / "pointwise_predictions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    outdir: Path = args.outdir or (input_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    run_config = _load_run_config(input_dir)
    trial_df = _load_trial_summary(input_dir)
    pointwise_df = _maybe_load_pointwise(input_dir)

    task = str(run_config["task"])
    metric = args.metric or ("mse" if task == "regression" else "error_rate")
    pred_col = args.pred_col or ("pred_score" if task == "regression" else "pred_prob")

    summary = build_analysis_summary(trial_df=trial_df, task_type=task, primary_metric=metric)
    save_json(summary, outdir / "analysis_summary.json")

    method_summary = pd.DataFrame(summary["method_summary"])
    save_table(method_summary, outdir / "method_summary_analysis.csv")

    if summary["subgroup_summary"]:
        subgroup_summary = pd.DataFrame(summary["subgroup_summary"])
        save_table(subgroup_summary, outdir / "subgroup_summary_analysis.csv")

    pairwise = make_pairwise_comparison_table(
        trial_df=trial_df,
        primary_metric=metric,
        baseline_method=args.pairwise_baseline,
    )
    save_table(pairwise, outdir / "pairwise_comparison.csv")

    make_method_comparison_plot(
        trial_df=trial_df,
        task_type=task,
        primary_metric=metric,
        outpath=outdir / "method_comparison.png",
    )
    make_error_variance_scatter(
        trial_df=trial_df,
        task_type=task,
        primary_metric=metric,
        outpath=outdir / "error_variance_scatter.png",
    )

    if pointwise_df is not None:
        pointwise_df.to_csv(outdir / "pointwise_predictions_preview.csv", index=False)

    if pointwise_df is not None and not args.skip_heatmaps:
        heatmap_dir = outdir / "slice_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        make_slice_heatmaps(
            pointwise_df=pointwise_df,
            task_type=task,
            pred_col=pred_col,
            outdir=heatmap_dir,
        )

    text_lines = [
        f"task={task}",
        f"primary_metric={metric}",
        f"n_methods={trial_df['method'].nunique()}",
        f"n_repetitions={trial_df['rep'].nunique()}",
        f"best_method={summary['best_method']}",
    ]
    (outdir / "analysis_notes.txt").write_text("\n".join(text_lines) + "\n")

    print(f"Wrote analysis summary to: {outdir / 'analysis_summary.json'}")
    print(f"Wrote pairwise comparison to: {outdir / 'pairwise_comparison.csv'}")
    print(f"Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()