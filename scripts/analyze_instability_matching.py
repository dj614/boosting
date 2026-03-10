#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from progress_utils import tqdm_iter
from sim.instability_matching_analysis import (
    build_analysis_summary,
    make_error_variance_scatter,
    make_method_comparison_plot,
    make_pairwise_comparison_table,
    make_slice_heatmaps,
    save_json,
    save_table,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze experiment 1 instability-matching outputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--baseline-method", type=str, default=None)
    return parser


def _primary_metric(task_type: str) -> str:
    return "mse" if task_type == "regression" else "error_rate"


def _subgroup_summary_frame(summary: Dict[str, object]) -> pd.DataFrame:
    rows = list(summary.get("subgroup_summary", []))
    if not rows:
        return pd.DataFrame(columns=["method", "group_metric", "mean", "se"])
    out = pd.DataFrame(rows)
    return out.sort_values(["group_metric", "mean", "method"]).reset_index(drop=True)


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    outdir = args.outdir or (input_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    trial_path = input_dir / "trial_metrics.csv"
    if not trial_path.exists():
        raise FileNotFoundError(f"Missing trial metrics file: {trial_path}")
    trial_df = pd.read_csv(trial_path)
    pointwise_path = input_dir / "pointwise_predictions.csv"
    pointwise_df = pd.read_csv(pointwise_path) if pointwise_path.exists() else None

    manifest: List[Dict[str, object]] = []
    combos = [
        (task_type, scenario)
        for task_type in sorted(trial_df["task_type"].dropna().astype(str).unique().tolist())
        for scenario in sorted(trial_df.loc[trial_df["task_type"] == task_type, "scenario"].dropna().astype(str).unique().tolist())
    ]
    for task_type, scenario in tqdm_iter(combos, total=len(combos), desc="Experiment 1 analysis", unit="combo"):
            subset = trial_df.loc[(trial_df["task_type"] == task_type) & (trial_df["scenario"] == scenario)].copy()
            if subset.empty:
                continue
            primary_metric = _primary_metric(task_type)
            combo_dir = outdir / f"{task_type}_{scenario}"
            combo_dir.mkdir(parents=True, exist_ok=True)

            summary = build_analysis_summary(subset, task_type=task_type, primary_metric=primary_metric)
            save_json(summary, combo_dir / "analysis_summary.json")

            method_summary = pd.DataFrame(summary.get("method_summary", []))
            subgroup_summary = _subgroup_summary_frame(summary)
            save_table(method_summary, combo_dir / "method_summary.csv")
            save_table(subgroup_summary, combo_dir / "subgroup_summary.csv")

            pairwise = make_pairwise_comparison_table(
                subset,
                primary_metric=primary_metric,
                baseline_method=args.baseline_method,
            )
            save_table(pairwise, combo_dir / "pairwise_comparison.csv")

            make_method_comparison_plot(
                subset,
                task_type=task_type,
                primary_metric=primary_metric,
                outpath=combo_dir / "method_comparison.png",
            )
            make_error_variance_scatter(
                subset,
                task_type=task_type,
                primary_metric=primary_metric,
                outpath=combo_dir / "error_variance_scatter.png",
            )

            if pointwise_df is not None:
                point_subset = pointwise_df.loc[
                    (pointwise_df["task_type"] == task_type) & (pointwise_df["scenario"] == scenario)
                ].copy()
                if not point_subset.empty:
                    make_slice_heatmaps(
                        point_subset,
                        task_type=task_type,
                        pred_col="pred_mean",
                        outdir=combo_dir,
                    )

            manifest.append(
                {
                    "task_type": task_type,
                    "scenario": scenario,
                    "primary_metric": primary_metric,
                    "best_method": summary.get("best_method"),
                    "output_dir": str(combo_dir),
                }
            )

    save_json({"runs": manifest}, outdir / "analysis_manifest.json")
    print(json.dumps({"runs": manifest}, indent=2))
    print(f"Wrote analysis outputs to: {outdir}")


if __name__ == "__main__":
    main()
