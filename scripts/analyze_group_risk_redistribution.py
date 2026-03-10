#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Dict,List
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.group_risk_redistribution_analysis import (  # noqa: E402
    FocusPair,
    aggregate_trajectories,
    bootstrap_pairwise_metric_differences,
    infer_focus_pairs,
    load_step2_artifacts,
    make_all_pairwise_seed_comparisons,
    make_analysis_summary,
    make_delta_loss_distribution_plot,
    make_group_risk_bars,
    make_group_risk_trajectory_plot,
    make_overall_vs_worst_group_scatter,
    make_worst_group_trajectory_plot,
    pairwise_loss_deltas,
    save_json,
    save_table,
    summarize_group_metrics,
    summarize_model_metrics,
)


KEY_METRICS = [
    "overall_log_loss",
    "core_worst_group_log_loss",
    "core_tail_log_loss_top_10pct",
    "core_group_log_loss_variance_weighted",
]

def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze experiment 2 group-risk redistribution outputs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory produced by scripts/run_group_risk_trajectory_benchmark.py",
    )
    parser.add_argument("--outdir", type=Path, default=None, help="Defaults to <input-dir>/analysis")
    parser.add_argument("--split", type=str, default="test", help="Which split to analyze. Default: test")
    parser.add_argument("--focus-baseline", type=str, default=None, help="Optional exact baseline model name for one focus pair")
    parser.add_argument("--focus-candidate", type=str, default=None, help="Optional exact candidate model name for one focus pair")
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--baseline-top-frac", type=float, default=0.10)
    return parser


def _resolve_focus_pairs(metrics_df: pd.DataFrame, baseline: str | None, candidate: str | None) -> List[FocusPair]:
    if baseline and candidate:
        return [FocusPair(baseline_model=baseline, candidate_model=candidate, label=f"{candidate}_vs_{baseline}")]
    model_names = sorted(metrics_df["model_name"].astype(str).unique().tolist()) if not metrics_df.empty else []
    return infer_focus_pairs(model_names)

def _pair_summary(
    *,
    pair: FocusPair,
    pairwise_seed_pair: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    group_delta_df: pd.DataFrame,
    merged_delta_df: pd.DataFrame,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "pair_label": pair.label,
        "baseline_model": pair.baseline_model,
        "candidate_model": pair.candidate_model,
        "n_samples_compared": int(merged_delta_df.shape[0]),
        "n_groups": int(group_delta_df["group"].nunique()) if not group_delta_df.empty else 0,
    }

    if not pairwise_seed_pair.empty:
        key_metrics = pairwise_seed_pair.loc[pairwise_seed_pair["metric"].isin(KEY_METRICS)].copy()
        summary["seed_level_deltas"] = key_metrics[
            ["metric", "delta_mean", "candidate_better_rate", "paired_t_pvalue", "wilcoxon_pvalue"]
        ].to_dict("records")

    if not bootstrap_df.empty:
        summary["bootstrap_deltas"] = bootstrap_df[
            ["metric", "delta_mean_bootstrap", "delta_ci_lower_95", "delta_ci_upper_95"]
        ].to_dict("records")

    if not group_delta_df.empty:
        worst_group_row = group_delta_df.sort_values("mean_delta_log_loss", ascending=False).iloc[0]
        best_group_row = group_delta_df.sort_values("mean_delta_log_loss", ascending=True).iloc[0]
        summary["worst_redistributed_group"] = {
            "group": str(worst_group_row["group"]),
            "mean_delta_log_loss": float(worst_group_row["mean_delta_log_loss"]),
        }
        summary["best_redistributed_group"] = {
            "group": str(best_group_row["group"]),
            "mean_delta_log_loss": float(best_group_row["mean_delta_log_loss"]),
        }
    return summary


def main() -> None:

    parser = _make_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    outdir = args.outdir or (input_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts = load_step2_artifacts(input_dir=input_dir, prediction_splits=[args.split])
    model_summary = summarize_model_metrics(artifacts.metrics_summary, split=args.split)
    group_summary = summarize_group_metrics(artifacts.group_metrics_summary, split=args.split)
    trajectory_core_agg, trajectory_group_agg = aggregate_trajectories(
        artifacts.trajectory_core,
        artifacts.trajectory_groups,
        split=args.split,
    )

    metrics_for_split = (
        artifacts.metrics_summary.loc[artifacts.metrics_summary["split"] == args.split].copy()
        if not artifacts.metrics_summary.empty
        else pd.DataFrame()
    )
    
    summary = make_analysis_summary(artifacts.metrics_summary, split=args.split)
    focus_pairs = _resolve_focus_pairs(metrics_for_split, args.focus_baseline, args.focus_candidate)
    summary["focus_pairs"] = [pair.__dict__ for pair in focus_pairs]
    save_json(summary, outdir / "analysis_summary.json")
    save_table(model_summary, outdir / "model_summary_analysis.csv")
    save_table(group_summary, outdir / "group_summary_analysis.csv")
    save_table(trajectory_core_agg, outdir / "trajectory_core_aggregated.csv")
    save_table(trajectory_group_agg, outdir / "trajectory_group_aggregated.csv")

    pairwise_seed = make_all_pairwise_seed_comparisons(
        artifacts.metrics_summary,
        split=args.split,
        focus_pairs=focus_pairs,
    )
    save_table(pairwise_seed, outdir / "pairwise_seed_comparisons.csv")

    if not model_summary.empty:
        make_overall_vs_worst_group_scatter(model_summary, figures_dir / "overall_vs_worst_group_scatter.png")

    bootstrap_frames: List[pd.DataFrame] = []
    group_delta_frames: List[pd.DataFrame] = []
    pair_manifest: List[Dict[str, object]] = []

    for pair in focus_pairs:
        pair_dir = outdir / pair.label
        pair_dir.mkdir(parents=True, exist_ok=True)

        pairwise_seed_pair = (
            pairwise_seed.loc[pairwise_seed["pair_label"] == pair.label].copy() if not pairwise_seed.empty else pd.DataFrame()
        )
        if not pairwise_seed_pair.empty:
            save_table(pairwise_seed_pair, pair_dir / "pairwise_seed_comparisons.csv")

        group_delta_df, merged_delta_df = pairwise_loss_deltas(
            artifacts.predictions,
            split=args.split,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            baseline_top_frac=args.baseline_top_frac,
       )
        if group_delta_df.empty or merged_delta_df.empty:
            pair_manifest.append(
                {
                    "pair_label": pair.label,
                    "baseline_model": pair.baseline_model,
                    "candidate_model": pair.candidate_model,
                    "status": "missing_pairwise_predictions",
                }
            )
            continue

        group_delta_df = group_delta_df.copy()
        group_delta_df.insert(0, "pair_label", pair.label)
        merged_delta_df = merged_delta_df.copy()
        merged_delta_df.insert(0, "pair_label", pair.label)
        save_table(group_delta_df, pair_dir / "group_loss_delta_summary.csv")
        save_table(merged_delta_df, pair_dir / "individual_loss_deltas.csv")
        group_delta_frames.append(group_delta_df)

        bootstrap_df = bootstrap_pairwise_metric_differences(
            merged_delta_df,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            n_bootstrap=args.bootstrap_iters,
        )
        if not bootstrap_df.empty:
            bootstrap_df = bootstrap_df.copy()
            bootstrap_df.insert(0, "pair_label", pair.label)
            save_table(bootstrap_df, pair_dir / "bootstrap_pairwise_metric_differences.csv")
            bootstrap_frames.append(bootstrap_df)

        make_group_risk_bars(
            group_summary,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            outpath=pair_dir / "group_risk_bars.png",
        )
        make_group_risk_trajectory_plot(
            trajectory_group_agg,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            outpath=pair_dir / "group_risk_trajectory.png",
        )
        make_worst_group_trajectory_plot(
            trajectory_core_agg,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            outpath=pair_dir / "worst_group_trajectory.png",
        )
        make_delta_loss_distribution_plot(
            merged_delta_df,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            outpath=pair_dir / "delta_log_loss_distribution.png",
        )

        pair_summary = _pair_summary(
            pair=pair,
            pairwise_seed_pair=pairwise_seed_pair,
            bootstrap_df=bootstrap_df,
            group_delta_df=group_delta_df,
            merged_delta_df=merged_delta_df,
        )
        save_json(pair_summary, pair_dir / "summary.json")
        pair_manifest.append(pair_summary)

    if bootstrap_frames:
        save_table(pd.concat(bootstrap_frames, ignore_index=True), outdir / "bootstrap_pairwise_metric_differences.csv")
    if group_delta_frames:
        save_table(pd.concat(group_delta_frames, ignore_index=True), outdir / "group_loss_delta_summary.csv")

    save_json({"pairs": pair_manifest}, outdir / "analysis_manifest.json")
    print(
        json.dumps(
            {
                "split": args.split,
                "n_focus_pairs": len(focus_pairs),
                "completed_pairs": sum(1 for item in pair_manifest if item.get("status") != "missing_pairwise_predictions"),
                "outdir": str(outdir),
            },
            indent=2,
        )
    )
    print(f"Wrote analysis outputs to: {outdir}")


if __name__ == "__main__":
    main()