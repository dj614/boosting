#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim.ctb_semantics import normalize_ctb_tree_family_name

PAIR_RE = re.compile(r"^(?P<family>[A-Za-z0-9_]+)_depth(?P<depth>\d+)")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze experiment 2 group-risk redistribution outputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--task-types", nargs="*", choices=["classification", "regression"], default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--baseline-top-frac", type=float, default=0.10)
    return parser


def _read_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _concat_csvs(paths: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(paths):
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _save_json(payload: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _normalize_family_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "family_name" not in df.columns:
        return df
    out = df.copy()
    out["family_name"] = out["family_name"].map(normalize_ctb_tree_family_name)
    return out


def _task_metric_spec(task_type: str) -> Dict[str, str]:
    if task_type == "classification":
        return {
            "primary": "overall_log_loss",
            "worst": "core_worst_group_log_loss",
            "group_metric": "log_loss",
            "sample_loss": "log_loss_i",
            "ylabel": "Log-loss",
        }
    return {
        "primary": "overall_mse",
        "worst": "core_worst_group_mse",
        "group_metric": "mse",
        "sample_loss": "squared_error_i",
        "ylabel": "MSE",
    }


def _parse_model_name(model_name: str) -> Tuple[str, int | None]:
    m = PAIR_RE.match(str(model_name))
    if m is None:
        return str(model_name).split("__", 1)[0], None
    return normalize_ctb_tree_family_name(str(m.group("family"))), int(m.group("depth"))


def _infer_focus_pairs(model_names: Sequence[str]) -> List[Tuple[str, str, str]]:
    parsed: Dict[Tuple[str, int], str] = {}
    for name in sorted({str(x) for x in model_names}):
        family, depth = _parse_model_name(name)
        if depth is not None:
            parsed[(family, depth)] = name
    pairs: List[Tuple[str, str, str]] = []
    for depth in sorted({d for _, d in parsed.keys()}):
        bag = parsed.get(("bagging", depth))
        gbdt = parsed.get(("gbdt", depth))
        if bag and gbdt:
            pairs.append((bag, gbdt, f"gbdt_vs_bagging_depth{depth}"))
        rf = parsed.get(("rf", depth))
        xgb = parsed.get(("xgb", depth))
        if rf and xgb:
            pairs.append((rf, xgb, f"xgb_vs_rf_depth{depth}"))
        ctb_candidates = [name for (family, d), name in parsed.items() if family == "ctb" and d == depth]
        if len(ctb_candidates) >= 2:
            legacy = next((n for n in ctb_candidates if "mode-legacy" in n), None)
            loss_aware = next((n for n in ctb_candidates if "mode-loss_aware" in n), None)
            if legacy and loss_aware:
                pairs.append((legacy, loss_aware, f"ctb_loss_aware_vs_legacy_depth{depth}"))
    return pairs


def _aggregate(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=list(group_cols))
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in set(group_cols)]
    mean_df = df.groupby(list(group_cols), dropna=False)[numeric_cols].mean().reset_index()
    se_df = df.groupby(list(group_cols), dropna=False)[numeric_cols].sem().reset_index().rename(columns={c: f"{c}_se" for c in numeric_cols})
    return mean_df.merge(se_df, on=list(group_cols), how="left")


def _scatter(model_summary: pd.DataFrame, *, x_col: str, y_col: str, title: str, outpath: Path) -> None:
    if model_summary.empty or x_col not in model_summary.columns or y_col not in model_summary.columns:
        return
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(model_summary[x_col], model_summary[y_col])
    for _, row in model_summary.iterrows():
        ax.annotate(str(row["model_name"]), (float(row[x_col]), float(row[y_col])), fontsize=7)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _group_bar(group_summary: pd.DataFrame, *, baseline: str, candidate: str, metric: str, ylabel: str, outpath: Path) -> None:
    sub = group_summary.loc[group_summary["model_name"].isin([baseline, candidate])].copy()
    if sub.empty or metric not in sub.columns:
        return
    groups = sorted(sub["group"].astype(str).unique().tolist())
    x = np.arange(len(groups), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for offset, model_name in [(-width / 2.0, baseline), (width / 2.0, candidate)]:
        mdf = sub.loc[sub["model_name"] == model_name].set_index("group").reindex(groups)
        mean = pd.to_numeric(mdf[metric], errors="coerce").to_numpy(dtype=float)
        se_col = f"{metric}_se"
        se = pd.to_numeric(mdf[se_col], errors="coerce").to_numpy(dtype=float) if se_col in mdf.columns else None
        ax.bar(x + offset, mean, width=width, yerr=se, label=model_name, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Group risk: {candidate} vs {baseline}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _trajectory_plot(df: pd.DataFrame, *, baseline: str, candidate: str, value_col: str, title: str, ylabel: str, outpath: Path, group_col: str | None = None) -> None:
    sub = df.loc[df["model_name"].isin([baseline, candidate])].copy()
    if sub.empty or value_col not in sub.columns:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if group_col is None:
        for model_name in [baseline, candidate]:
            mdf = sub.loc[sub["model_name"] == model_name].sort_values("checkpoint")
            ax.plot(mdf["checkpoint"], mdf[value_col], label=model_name)
    else:
        for model_name, linestyle in [(baseline, "--"), (candidate, "-")]:
            for group_name in sorted(sub[group_col].astype(str).unique().tolist()):
                gdf = sub.loc[(sub["model_name"] == model_name) & (sub[group_col] == group_name)].sort_values("checkpoint")
                if not gdf.empty:
                    ax.plot(gdf["checkpoint"], gdf[value_col], linestyle=linestyle, label=f"{model_name}:{group_name}")
    ax.set_xlabel("Ensemble size / iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _pairwise_loss_deltas(predictions_df: pd.DataFrame, *, split: str, baseline_model: str, candidate_model: str, sample_loss_col: str, baseline_top_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = predictions_df.loc[predictions_df["split"] == split].copy()
    base_cols = ["dataset_name", "seed", "sample_id", "group", "y_true", sample_loss_col]
    base = sub.loc[sub["model_name"] == baseline_model, base_cols].copy()
    cand = sub.loc[sub["model_name"] == candidate_model, base_cols].copy()
    if base.empty or cand.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = base.merge(cand, on=["dataset_name", "seed", "sample_id", "group", "y_true"], suffixes=("_baseline", "_candidate"))
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged["delta_loss"] = merged[f"{sample_loss_col}_candidate"] - merged[f"{sample_loss_col}_baseline"]
    merged["candidate_better"] = (merged["delta_loss"] < 0.0).astype(int)
    merged["baseline_top_loss"] = 0
    ranked_frames: List[pd.DataFrame] = []
    for (_, seed), gdf in merged.groupby(["dataset_name", "seed"], dropna=False):
        local = gdf.copy()
        n = local.shape[0]
        k = max(1, int(math.ceil(float(baseline_top_frac) * n)))
        idx = np.argsort(local[f"{sample_loss_col}_baseline"].to_numpy(dtype=float))[::-1][:k]
        top_mask = np.zeros(n, dtype=int)
        top_mask[idx] = 1
        local["baseline_top_loss"] = top_mask
        ranked_frames.append(local)
    merged = pd.concat(ranked_frames, ignore_index=True)
    rows: List[Dict[str, object]] = []
    for group_name, gdf in merged.groupby("group", dropna=False):
        top_df = gdf.loc[gdf["baseline_top_loss"] == 1]
        rows.append(
            {
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "group": str(group_name),
                "n": int(gdf.shape[0]),
                "mean_delta_loss": float(gdf["delta_loss"].mean()),
                "median_delta_loss": float(gdf["delta_loss"].median()),
                "candidate_better_rate": float(gdf["candidate_better"].mean()),
                "n_baseline_top_loss": int(top_df.shape[0]),
                "mean_delta_loss_on_baseline_top_loss": float(top_df["delta_loss"].mean()) if not top_df.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True), merged


def _delta_boxplot(merged_delta_df: pd.DataFrame, *, ylabel: str, title: str, outpath: Path) -> None:
    if merged_delta_df.empty:
        return
    groups = sorted(merged_delta_df["group"].astype(str).unique().tolist())
    data = [merged_delta_df.loc[merged_delta_df["group"] == group, "delta_loss"].to_numpy(dtype=float) for group in groups]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.boxplot(data, labels=groups, vert=True, showfliers=False)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    outdir = args.outdir or (input_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_summary = _normalize_family_column(_read_if_exists(input_dir / "metrics_summary.csv"))
    metrics_summary_family = _normalize_family_column(_read_if_exists(input_dir / "metrics_summary_family_selected.csv"))
    group_metrics_summary = _normalize_family_column(_read_if_exists(input_dir / "group_metrics_summary.csv"))
    group_metrics_family = _normalize_family_column(_read_if_exists(input_dir / "group_metrics_summary_family_selected.csv"))
    trajectory_core = _concat_csvs(list(input_dir.glob("**/trajectory_core.csv")))
    trajectory_groups = _concat_csvs(list(input_dir.glob("**/trajectory_groups.csv")))
    predictions = _concat_csvs(list(input_dir.glob(f"**/predictions_{args.split}.csv")))

    if metrics_summary.empty:
        raise FileNotFoundError(f"Missing metrics_summary.csv under {input_dir}")

    available_tasks = sorted(metrics_summary["task_type"].dropna().astype(str).unique().tolist())
    task_types = args.task_types or available_tasks
    manifest: List[Dict[str, object]] = []

    for task_type in task_types:
        spec = _task_metric_spec(task_type)
        task_dir = outdir / task_type
        task_dir.mkdir(parents=True, exist_ok=True)

        metric_df = metrics_summary.loc[(metrics_summary["task_type"] == task_type) & (metrics_summary["split"] == args.split)].copy()
        group_df = group_metrics_summary.loc[(group_metrics_summary["task_type"] == task_type) & (group_metrics_summary["split"] == args.split)].copy()
        core_df = trajectory_core.loc[(trajectory_core["task_type"] == task_type) & (trajectory_core["split"] == args.split)].copy() if not trajectory_core.empty else pd.DataFrame()
        traj_group_df = trajectory_groups.loc[(trajectory_groups["task_type"] == task_type) & (trajectory_groups["split"] == args.split)].copy() if not trajectory_groups.empty else pd.DataFrame()
        pred_df = predictions.loc[predictions["task_type"] == task_type].copy() if not predictions.empty else pd.DataFrame()

        model_summary = _aggregate(metric_df, ["task_type", "dataset_name", "family_name", "model_name"])
        group_summary = _aggregate(group_df, ["task_type", "dataset_name", "family_name", "model_name", "group"])
        _save_table(model_summary, task_dir / "model_summary_analysis.csv")
        _save_table(group_summary, task_dir / "group_summary_analysis.csv")

        if not metrics_summary_family.empty:
            family_df = metrics_summary_family.loc[(metrics_summary_family["task_type"] == task_type) & (metrics_summary_family["split"] == args.split)].copy()
            family_summary = _aggregate(family_df, ["task_type", "dataset_name", "family_name", "model_name"])
            _save_table(family_summary, task_dir / "model_summary_family_selected.csv")
        if not group_metrics_family.empty:
            group_family_df = group_metrics_family.loc[(group_metrics_family["task_type"] == task_type) & (group_metrics_family["split"] == args.split)].copy()
            group_family_summary = _aggregate(group_family_df, ["task_type", "dataset_name", "family_name", "model_name", "group"])
            _save_table(group_family_summary, task_dir / "group_summary_family_selected.csv")

        if not model_summary.empty:
            _scatter(
                model_summary,
                x_col=f"{spec['primary']}",
                y_col=f"{spec['worst']}",
                title=f"{task_type}: overall vs worst-group",
                outpath=task_dir / "overall_vs_worst_group_scatter.png",
            )

        focus_pairs = _infer_focus_pairs(metric_df["model_name"].astype(str).unique().tolist())
        pair_rows: List[Dict[str, object]] = []
        group_delta_frames: List[pd.DataFrame] = []
        for baseline_model, candidate_model, label in focus_pairs:
            pair_dir = task_dir / label
            pair_dir.mkdir(parents=True, exist_ok=True)
            group_delta_df, merged_delta_df = _pairwise_loss_deltas(
                pred_df,
                split=args.split,
                baseline_model=baseline_model,
                candidate_model=candidate_model,
                sample_loss_col=spec["sample_loss"],
                baseline_top_frac=args.baseline_top_frac,
            )
            if not group_delta_df.empty:
                _save_table(group_delta_df, pair_dir / "group_loss_delta_summary.csv")
                _save_table(merged_delta_df, pair_dir / "individual_loss_deltas.csv")
                group_delta_frames.append(group_delta_df.assign(pair_label=label))
                _delta_boxplot(
                    merged_delta_df,
                    ylabel=f"candidate - baseline {spec['ylabel']}",
                    title=f"{candidate_model} vs {baseline_model}",
                    outpath=pair_dir / "delta_loss_distribution.png",
                )
                _group_bar(
                    group_summary,
                    baseline=baseline_model,
                    candidate=candidate_model,
                    metric=spec["group_metric"],
                    ylabel=f"Group {spec['ylabel']}",
                    outpath=pair_dir / "group_risk_bars.png",
                )
                _trajectory_plot(
                    core_df,
                    baseline=baseline_model,
                    candidate=candidate_model,
                    value_col=spec["worst"],
                    title=f"Worst-group trajectory: {candidate_model} vs {baseline_model}",
                    ylabel=f"Worst-group {spec['ylabel']}",
                    outpath=pair_dir / "worst_group_trajectory.png",
                )
                _trajectory_plot(
                    traj_group_df,
                    baseline=baseline_model,
                    candidate=candidate_model,
                    value_col=spec["group_metric"],
                    title=f"Group trajectory: {candidate_model} vs {baseline_model}",
                    ylabel=f"Group {spec['ylabel']}",
                    outpath=pair_dir / "group_risk_trajectory.png",
                    group_col="group",
                )
            pair_rows.append({"pair_label": label, "baseline_model": baseline_model, "candidate_model": candidate_model, "has_pairwise_predictions": int(not group_delta_df.empty)})
        _save_table(pd.DataFrame(pair_rows), task_dir / "focus_pairs.csv")
        if group_delta_frames:
            _save_table(pd.concat(group_delta_frames, ignore_index=True), task_dir / "group_loss_delta_summary.csv")

        summary_payload = {
            "task_type": task_type,
            "split": args.split,
            "n_models": int(metric_df["model_name"].nunique()) if not metric_df.empty else 0,
            "models": sorted(metric_df["model_name"].astype(str).unique().tolist()) if not metric_df.empty else [],
            "focus_pairs": pair_rows,
        }
        _save_json(summary_payload, task_dir / "analysis_summary.json")
        manifest.append(summary_payload)

    _save_json({"tasks": manifest}, outdir / "analysis_manifest.json")
    print(json.dumps({"tasks": manifest}, indent=2))
    print(f"Wrote analysis outputs to: {outdir}")


if __name__ == "__main__":
    main()
