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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from progress_utils import tqdm_iter
from sim.sparse_recovery_eval import aggregate_metric_table, make_feature_support_frame, stability_selection_metrics


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze experiment 4 sparse recovery outputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--top-features", type=int, default=25)
    return parser


def _save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _parse_supports(series: pd.Series) -> List[np.ndarray]:
    supports: List[np.ndarray] = []
    for value in series.fillna("[]"):
        parsed = json.loads(str(value))
        supports.append(np.asarray(parsed, dtype=int))
    return supports


def _stability_frame(trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    groups = list(trial_df.groupby(["task_type", "design", "family_name"], dropna=False))
    for (task_type, design, family_name), sub_df in tqdm_iter(groups, total=len(groups), desc="Feature frequencies", unit="group"):
        p = int(sub_df["p"].iloc[0])
        stability = stability_selection_metrics(_parse_supports(sub_df["support_hat_json"]), p=p)
        rows.append(
            {
                "task_type": task_type,
                "design": design,
                "family_name": family_name,
                "n_supports": stability["n_supports"],
                "mean_support_size": stability["mean_support_size"],
                "support_size_se": stability["support_size_se"],
                "pairwise_jaccard_mean": stability["pairwise_jaccard_mean"],
                "pairwise_jaccard_se": stability["pairwise_jaccard_se"],
            }
        )
    return pd.DataFrame(rows).sort_values(["task_type", "design", "family_name"]).reset_index(drop=True)


def _feature_frequency_tables(trial_df: pd.DataFrame, outdir: Path, top_features: int) -> List[Dict[str, object]]:
    manifest: List[Dict[str, object]] = []
    feature_dir = outdir / "feature_frequency_tables"
    for (task_type, design, family_name), sub_df in trial_df.groupby(["task_type", "design", "family_name"], dropna=False):
        p = int(sub_df["p"].iloc[0])
        supports = _parse_supports(sub_df["support_hat_json"])
        frame = make_feature_support_frame(p=p, supports=supports)
        frame = frame.sort_values(["selection_frequency", "feature_idx"], ascending=[False, True]).reset_index(drop=True)
        path = feature_dir / f"{task_type}__{design}__{family_name}.csv"
        _save_table(frame, path)
        manifest.append(
            {
                "task_type": task_type,
                "design": design,
                "family_name": family_name,
                "path": str(path),
                "top_features": frame.head(top_features).to_dict(orient="records"),
            }
        )
    return manifest


def _task_primary_columns(task_type: str) -> tuple[str, str, str]:
    if task_type == "classification":
        return "valid_log_loss", "test_log_loss", "support_f1"
    return "valid_mse", "test_mse", "support_f1"


def _plot_metric_bar(summary_df: pd.DataFrame, task_type: str, design: str, metric_col: str, ylabel: str, outpath: Path) -> None:
    sub_df = summary_df.loc[(summary_df["task_type"] == task_type) & (summary_df["design"] == design)].copy()
    if sub_df.empty or metric_col not in sub_df.columns:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(sub_df["family_name"].astype(str), sub_df[metric_col].astype(float))
    ax.set_xlabel("Model family")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{task_type} / {design}: {ylabel}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _plot_prediction_vs_recovery(summary_df: pd.DataFrame, task_type: str, design: str, pred_col: str, outpath: Path) -> None:
    sub_df = summary_df.loc[(summary_df["task_type"] == task_type) & (summary_df["design"] == design)].copy()
    if sub_df.empty or pred_col not in sub_df.columns or "support_f1_mean" not in sub_df.columns:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(sub_df[pred_col], sub_df["support_f1_mean"])
    for _, row in sub_df.iterrows():
        ax.annotate(str(row["family_name"]), (float(row[pred_col]), float(row["support_f1_mean"])), fontsize=8)
    ax.set_xlabel(pred_col)
    ax.set_ylabel("Mean support F1")
    ax.set_title(f"{task_type} / {design}: prediction vs structure recovery")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    outdir: Path = args.outdir or (input_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    family_selected_path = input_dir / "trial_metrics_family_selected.csv"
    trial_path = family_selected_path if family_selected_path.exists() else (input_dir / "trial_metrics.csv")
    if not trial_path.exists():
        trial_path = input_dir / "trial_metrics.csv"
    if not trial_path.exists():
        raise FileNotFoundError(f"Missing trial metrics file: {trial_path}")
    trial_df = pd.read_csv(trial_path)
    if trial_df.empty:
        raise ValueError(f"{trial_path.name} is empty")

    summary_df = aggregate_metric_table(
        trial_df,
        group_cols=["task_type", "design", "family_name", "model_name", "support_eval_mode"],
        sort_by="test_mse_mean" if "test_mse" in trial_df.columns else "test_log_loss_mean",
        ascending=True,
    )
    stability_df = _stability_frame(trial_df)
    merged_df = summary_df.merge(stability_df, on=["task_type", "design", "family_name"], how="left")
    _save_table(summary_df, outdir / "model_summary.csv")
    _save_table(stability_df, outdir / "stability_summary.csv")
    _save_table(merged_df, outdir / "merged_summary.csv")


    feature_manifest = _feature_frequency_tables(trial_df, outdir=outdir, top_features=args.top_features)

    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    combos = sorted({(str(r.task_type), str(r.design)) for _, r in merged_df[["task_type", "design"]].drop_duplicates().iterrows()})
    for task_type, design in tqdm_iter(combos, total=len(combos), desc="Sparse recovery plots", unit="combo"):
        valid_col, test_col, _ = _task_primary_columns(task_type)
        ylabel = "Mean test log-loss" if task_type == "classification" else "Mean test MSE"
        _plot_metric_bar(merged_df, task_type, design, f"{test_col}_mean", ylabel, plot_dir / f"{task_type}__{design}_primary.png")
        _plot_metric_bar(merged_df, task_type, design, "support_f1_mean", "Mean support F1", plot_dir / f"{task_type}__{design}_support_f1.png")
        _plot_metric_bar(merged_df, task_type, design, "pairwise_jaccard_mean", "Support stability (Jaccard)", plot_dir / f"{task_type}__{design}_stability.png")
        _plot_prediction_vs_recovery(merged_df, task_type, design, f"{test_col}_mean", plot_dir / f"{task_type}__{design}_prediction_vs_recovery.png")
 
    manifest_rows: List[Dict[str, object]] = []
    for task_type, design in combos:
        design_df = merged_df.loc[(merged_df["task_type"] == task_type) & (merged_df["design"] == design)].copy()
        if design_df.empty:
            continue
        _, test_col, _ = _task_primary_columns(task_type)
        best_pred = design_df.sort_values(f"{test_col}_mean", ascending=True).iloc[0]
        best_struct = design_df.sort_values(["support_f1_mean", "pairwise_jaccard_mean"], ascending=[False, False]).iloc[0]
        manifest_rows.append(
            {
                "task_type": task_type,
                "design": design,
                "best_predictive_family": str(best_pred["family_name"]),
                "best_predictive_model": str(best_pred["model_name"]),
                f"best_predictive_{test_col}": float(best_pred[f"{test_col}_mean"]),
                "best_structural_family": str(best_struct["family_name"]),
                "best_structural_model": str(best_struct["model_name"]),
                "best_structural_support_f1": float(best_struct["support_f1_mean"]),
            }
        )

    summary_payload = {
        "trial_metrics_source": str(trial_path.name),
        "n_trials": float(len(trial_df)),
        "task_types": sorted(trial_df["task_type"].dropna().astype(str).unique().tolist()),
        "designs": sorted(trial_df["design"].dropna().astype(str).unique().tolist()),
        "families": sorted(trial_df["family_name"].dropna().astype(str).unique().tolist()) if "family_name" in trial_df.columns else [],
        "families": sorted(trial_df["family_name"].dropna().astype(str).unique().tolist()),
        "design_summary": manifest_rows,
        "feature_frequency_manifest": feature_manifest,
    }
    _save_json(summary_payload, outdir / "analysis_summary.json")
    print(json.dumps(summary_payload, indent=2))
    print(f"Wrote analysis outputs to: {outdir}")


if __name__ == "__main__":
    main()
