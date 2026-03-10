#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.experiment1_step3_analysis import (  # noqa: E402
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


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze experiment-1 step-2 outputs for risk redistribution.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory produced by scripts/run_experiment1_step2.py")
    parser.add_argument("--outdir", type=Path, default=None, help="Defaults to <input-dir>/analysis_step3")
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



def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    outdir = args.outdir or (input_dir / "analysis_step3")
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

    summary = make_analysis_summary(artifacts.metrics_summary, split=args.split)
    focus_pairs = _resolve_focus_pairs(artifacts.metrics_summary.loc[artifacts.metrics_summary["split"] == args.split].copy(), args.focus_baseline, args.focus_candidate)
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

    bootstrap_frames = []
    group_delta_frames = []
    for pair in focus_pairs:
        pair_dir = outdir / pair.label
        pair_dir.mkdir(parents=True, exist_ok=True)

        group_delta_df, merged_delta_df = pairwise_loss_deltas(
            artifacts.predictions,
            split=args.split,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            baseline_top_frac=float(args.baseline_top_frac),
        )
        save_table(group_delta_df, pair_dir / "pairwise_group_delta.csv")
        if not merged_delta_df.empty:
            save_table(merged_delta_df, pair_dir / "sample_loss_deltas.csv")
            bootstrap_df = bootstrap_pairwise_metric_differences(
                merged_delta_df,
                baseline_model=pair.baseline_model,
                candidate_model=pair.candidate_model,
                n_bootstrap=int(args.bootstrap_iters),
            )
            bootstrap_df.insert(0, "pair_label", pair.label)
            bootstrap_frames.append(bootstrap_df)
            make_delta_loss_distribution_plot(
                merged_delta_df,
                baseline_model=pair.baseline_model,
                candidate_model=pair.candidate_model,
                outpath=figures_dir / f"delta_loss_distribution_{pair.label}.png",
            )
        if not group_delta_df.empty:
            group_delta_df.insert(0, "pair_label", pair.label)
            group_delta_frames.append(group_delta_df)

        make_group_risk_bars(
            group_summary,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            outpath=figures_dir / f"group_risk_bars_{pair.label}.png",
        )
        make_group_risk_trajectory_plot(
            trajectory_group_agg,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            outpath=figures_dir / f"group_risk_trajectory_{pair.label}.png",
        )
        make_worst_group_trajectory_plot(
            trajectory_core_agg,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
            outpath=figures_dir / f"worst_group_trajectory_{pair.label}.png",
        )

    if bootstrap_frames:
        save_table(pd.concat(bootstrap_frames, ignore_index=True), outdir / "bootstrap_pairwise_differences.csv")
    else:
        save_table(pd.DataFrame(), outdir / "bootstrap_pairwise_differences.csv")

    if group_delta_frames:
        save_table(pd.concat(group_delta_frames, ignore_index=True), outdir / "pairwise_group_deltas_all.csv")
    else:
        save_table(pd.DataFrame(), outdir / "pairwise_group_deltas_all.csv")

    notes = [
        f"split={args.split}",
        f"n_models={artifacts.metrics_summary.loc[artifacts.metrics_summary['split'] == args.split, 'model_name'].nunique() if not artifacts.metrics_summary.empty else 0}",
        f"n_focus_pairs={len(focus_pairs)}",
        f"bootstrap_iters={int(args.bootstrap_iters)}",
    ]
    (outdir / "analysis_notes.txt").write_text("\n".join(notes) + "\n")

    print(f"Wrote analysis summary to: {outdir / 'analysis_summary.json'}")
    print(f"Wrote pairwise seed table to: {outdir / 'pairwise_seed_comparisons.csv'}")
    print(f"Wrote figures to: {figures_dir}")


if __name__ == "__main__":
    main()
