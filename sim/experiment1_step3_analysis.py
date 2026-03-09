from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # pragma: no cover
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover
    ttest_rel = None
    wilcoxon = None


Array = np.ndarray
PAIR_RE = re.compile(r"^(?P<family>[A-Za-z0-9_]+)_depth(?P<depth>\d+)$")
LOSS_METRICS = [
    "overall_log_loss",
    "core_worst_group_log_loss",
    "core_group_log_loss_variance_weighted",
    "core_tail_log_loss_top_10pct",
]


@dataclass(frozen=True)
class FocusPair:
    baseline_model: str
    candidate_model: str
    label: str


@dataclass
class Step2Artifacts:
    metrics_summary: pd.DataFrame
    group_metrics_summary: pd.DataFrame
    predictions: pd.DataFrame
    trajectory_core: pd.DataFrame
    trajectory_groups: pd.DataFrame


def _to_serializable(value: object) -> object:
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value



def save_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(payload, default=_to_serializable))
    path.write_text(json.dumps(serializable, indent=2, sort_keys=False))



def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



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



def load_step2_artifacts(input_dir: Path, prediction_splits: Sequence[str] = ("test",)) -> Step2Artifacts:
    metrics_summary = _read_if_exists(input_dir / "metrics_summary.csv")
    group_metrics_summary = _read_if_exists(input_dir / "group_metrics_summary.csv")

    prediction_paths: List[Path] = []
    for split in prediction_splits:
        prediction_paths.extend(input_dir.glob(f"**/predictions_{split}.csv"))
    predictions = _concat_csvs(prediction_paths)

    trajectory_core = _concat_csvs(list(input_dir.glob("**/trajectory_core.csv")))
    trajectory_groups = _concat_csvs(list(input_dir.glob("**/trajectory_groups.csv")))
    return Step2Artifacts(
        metrics_summary=metrics_summary,
        group_metrics_summary=group_metrics_summary,
        predictions=predictions,
        trajectory_core=trajectory_core,
        trajectory_groups=trajectory_groups,
    )



def summarize_model_metrics(metrics_df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    sub = metrics_df.loc[metrics_df["split"] == split].copy()
    if sub.empty:
        return pd.DataFrame()
    value_cols = [
        col
        for col in sub.columns
        if col.startswith("overall_") or col.startswith("core_") or col == "selected_checkpoint"
    ]
    grouped = sub.groupby(["dataset_name", "model_name"], dropna=False)
    mean_df = grouped[value_cols].mean().reset_index()
    se_df = grouped[value_cols].sem().reset_index().rename(columns={c: f"{c}_se" for c in value_cols})
    out = mean_df.merge(se_df, on=["dataset_name", "model_name"], how="left")
    if "overall_log_loss" in out.columns:
        out = out.sort_values(["dataset_name", "overall_log_loss", "core_worst_group_log_loss"]).reset_index(drop=True)
    return out



def summarize_group_metrics(group_df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    if group_df.empty:
        return pd.DataFrame()
    sub = group_df.loc[group_df["split"] == split].copy()
    if sub.empty:
        return pd.DataFrame()
    value_cols = [
        col
        for col in ["n", "positive_rate", "accuracy", "balanced_accuracy", "log_loss", "brier", "calibration_error", "roc_auc"]
        if col in sub.columns
    ]
    grouped = sub.groupby(["dataset_name", "model_name", "group"], dropna=False)
    mean_df = grouped[value_cols].mean().reset_index()
    se_df = grouped[value_cols].sem().reset_index().rename(columns={c: f"{c}_se" for c in value_cols})
    out = mean_df.merge(se_df, on=["dataset_name", "model_name", "group"], how="left")
    return out.sort_values(["dataset_name", "model_name", "group"]).reset_index(drop=True)



def _parse_model_name(model_name: str) -> Optional[Tuple[str, int]]:
    m = PAIR_RE.match(str(model_name))
    if m is None:
        return None
    return str(m.group("family")), int(m.group("depth"))



def infer_focus_pairs(model_names: Sequence[str]) -> List[FocusPair]:
    parsed: Dict[Tuple[str, int], str] = {}
    for name in sorted({str(x) for x in model_names}):
        key = _parse_model_name(name)
        if key is not None:
            parsed[key] = name

    out: List[FocusPair] = []
    for depth in sorted({depth for _, depth in parsed.keys()}):
        bag = parsed.get(("bagging", depth))
        gbdt = parsed.get(("gbdt", depth))
        if bag and gbdt:
            out.append(FocusPair(baseline_model=bag, candidate_model=gbdt, label=f"gbdt_vs_bagging_depth{depth}"))
        rf = parsed.get(("rf", depth))
        xgb = parsed.get(("xgb", depth))
        if rf and xgb:
            out.append(FocusPair(baseline_model=rf, candidate_model=xgb, label=f"xgb_vs_rf_depth{depth}"))

    if out:
        return out

    sorted_names = sorted({str(x) for x in model_names})
    if len(sorted_names) >= 2:
        return [FocusPair(baseline_model=sorted_names[0], candidate_model=sorted_names[1], label=f"{sorted_names[1]}_vs_{sorted_names[0]}")]
    return []



def make_analysis_summary(metrics_df: pd.DataFrame, split: str = "test") -> Dict[str, object]:
    if metrics_df.empty:
        return {"split": split, "available_models": [], "best_overall_log_loss_model": None, "best_worst_group_model": None}

    sub = metrics_df.loc[metrics_df["split"] == split].copy()
    model_summary = summarize_model_metrics(sub, split=split)
    if model_summary.empty:
        return {"split": split, "available_models": sorted(sub["model_name"].astype(str).unique().tolist())}

    best_overall_idx = model_summary["overall_log_loss"].astype(float).idxmin()
    best_worst_idx = model_summary["core_worst_group_log_loss"].astype(float).idxmin()
    return {
        "split": split,
        "available_models": sorted(sub["model_name"].astype(str).unique().tolist()),
        "n_seeds": int(sub["seed"].nunique()),
        "best_overall_log_loss_model": str(model_summary.loc[best_overall_idx, "model_name"]),
        "best_worst_group_model": str(model_summary.loc[best_worst_idx, "model_name"]),
        "focus_pairs": [pair.__dict__ for pair in infer_focus_pairs(sub["model_name"].astype(str).unique().tolist())],
    }



def paired_seed_comparison(
    metrics_df: pd.DataFrame,
    *,
    split: str,
    baseline_model: str,
    candidate_model: str,
    metric_columns: Sequence[str] = LOSS_METRICS,
) -> pd.DataFrame:
    sub = metrics_df.loc[metrics_df["split"] == split].copy()
    left = sub.loc[sub["model_name"] == baseline_model, ["dataset_name", "seed", *metric_columns]].copy()
    right = sub.loc[sub["model_name"] == candidate_model, ["dataset_name", "seed", *metric_columns]].copy()
    if left.empty or right.empty:
        return pd.DataFrame()
    merged = left.merge(right, on=["dataset_name", "seed"], suffixes=("_baseline", "_candidate"))
    if merged.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for metric in metric_columns:
        delta = merged[f"{metric}_candidate"].astype(float) - merged[f"{metric}_baseline"].astype(float)
        delta = delta.dropna()
        if delta.empty:
            continue
        arr = delta.to_numpy(dtype=float)
        row: Dict[str, object] = {
            "baseline_model": baseline_model,
            "candidate_model": candidate_model,
            "metric": metric,
            "n_pairs": int(arr.shape[0]),
            "delta_mean": float(np.mean(arr)),
            "delta_std": float(np.std(arr, ddof=1)) if arr.shape[0] > 1 else 0.0,
            "delta_se": float(np.std(arr, ddof=1) / math.sqrt(arr.shape[0])) if arr.shape[0] > 1 else 0.0,
            "delta_median": float(np.median(arr)),
            "candidate_better_rate": float(np.mean(arr < 0.0)),
        }
        if ttest_rel is not None and arr.shape[0] >= 2:
            baseline_vals = merged[f"{metric}_baseline"].astype(float).to_numpy(dtype=float)
            candidate_vals = merged[f"{metric}_candidate"].astype(float).to_numpy(dtype=float)
            try:
                row["paired_t_pvalue"] = float(ttest_rel(candidate_vals, baseline_vals, nan_policy="omit").pvalue)
            except Exception:
                row["paired_t_pvalue"] = float("nan")
        else:
            row["paired_t_pvalue"] = float("nan")
        if wilcoxon is not None and arr.shape[0] >= 1 and not np.allclose(arr, 0.0):
            try:
                row["wilcoxon_pvalue"] = float(wilcoxon(arr).pvalue)
            except Exception:
                row["wilcoxon_pvalue"] = float("nan")
        else:
            row["wilcoxon_pvalue"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)



def make_all_pairwise_seed_comparisons(
    metrics_df: pd.DataFrame,
    *,
    split: str,
    focus_pairs: Sequence[FocusPair],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for pair in focus_pairs:
        frame = paired_seed_comparison(
            metrics_df,
            split=split,
            baseline_model=pair.baseline_model,
            candidate_model=pair.candidate_model,
        )
        if not frame.empty:
            frame.insert(0, "pair_label", pair.label)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)



def pairwise_loss_deltas(
    predictions_df: pd.DataFrame,
    *,
    split: str,
    baseline_model: str,
    candidate_model: str,
    baseline_top_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sub = predictions_df.loc[predictions_df["split"] == split].copy()
    base_cols = ["dataset_name", "seed", "sample_id", "group", "y_true", "log_loss_i", "brier_i", "margin"]
    base = sub.loc[sub["model_name"] == baseline_model, base_cols].copy()
    cand = sub.loc[sub["model_name"] == candidate_model, base_cols].copy()
    if base.empty or cand.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged = base.merge(
        cand,
        on=["dataset_name", "seed", "sample_id", "group", "y_true"],
        suffixes=("_baseline", "_candidate"),
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged["delta_log_loss"] = merged["log_loss_i_candidate"] - merged["log_loss_i_baseline"]
    merged["delta_brier"] = merged["brier_i_candidate"] - merged["brier_i_baseline"]
    merged["candidate_better"] = (merged["delta_log_loss"] < 0.0).astype(int)
    merged["baseline_top_loss"] = 0

    ranked_frames: List[pd.DataFrame] = []
    for (_, seed), gdf in merged.groupby(["dataset_name", "seed"], dropna=False):
        local = gdf.copy()
        n = local.shape[0]
        k = max(1, int(math.ceil(float(baseline_top_frac) * n)))
        idx = np.argsort(local["log_loss_i_baseline"].to_numpy(dtype=float))[::-1][:k]
        top_mask = np.zeros(n, dtype=int)
        top_mask[idx] = 1
        local["baseline_top_loss"] = top_mask
        ranked_frames.append(local)
    merged = pd.concat(ranked_frames, ignore_index=True)

    rows: List[Dict[str, object]] = []
    for group_name, gdf in merged.groupby("group", dropna=False):
        row: Dict[str, object] = {
            "baseline_model": baseline_model,
            "candidate_model": candidate_model,
            "group": str(group_name),
            "n": int(gdf.shape[0]),
            "mean_delta_log_loss": float(gdf["delta_log_loss"].mean()),
            "median_delta_log_loss": float(gdf["delta_log_loss"].median()),
            "candidate_better_rate": float(gdf["candidate_better"].mean()),
        }
        top_df = gdf.loc[gdf["baseline_top_loss"] == 1]
        row["n_baseline_top_loss"] = int(top_df.shape[0])
        row["mean_delta_log_loss_on_baseline_top_loss"] = float(top_df["delta_log_loss"].mean()) if not top_df.empty else float("nan")
        row["candidate_better_rate_on_baseline_top_loss"] = float(top_df["candidate_better"].mean()) if not top_df.empty else float("nan")
        rows.append(row)

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True), merged



def aggregate_trajectories(trajectory_core_df: pd.DataFrame, trajectory_group_df: pd.DataFrame, *, split: str = "test") -> Tuple[pd.DataFrame, pd.DataFrame]:
    core = trajectory_core_df.loc[trajectory_core_df["split"] == split].copy() if not trajectory_core_df.empty else pd.DataFrame()
    group = trajectory_group_df.loc[trajectory_group_df["split"] == split].copy() if not trajectory_group_df.empty else pd.DataFrame()

    core_agg = pd.DataFrame()
    if not core.empty:
        value_cols = [c for c in core.columns if c.startswith("overall_") or c.startswith("core_")]
        grouped = core.groupby(["dataset_name", "model_name", "checkpoint"], dropna=False)
        mean_df = grouped[value_cols].mean().reset_index()
        se_df = grouped[value_cols].sem().reset_index().rename(columns={c: f"{c}_se" for c in value_cols})
        core_agg = mean_df.merge(se_df, on=["dataset_name", "model_name", "checkpoint"], how="left")

    group_agg = pd.DataFrame()
    if not group.empty:
        value_cols = [c for c in ["n", "positive_rate", "accuracy", "balanced_accuracy", "log_loss", "brier", "calibration_error", "roc_auc"] if c in group.columns]
        grouped = group.groupby(["dataset_name", "model_name", "checkpoint", "group"], dropna=False)
        mean_df = grouped[value_cols].mean().reset_index()
        se_df = grouped[value_cols].sem().reset_index().rename(columns={c: f"{c}_se" for c in value_cols})
        group_agg = mean_df.merge(se_df, on=["dataset_name", "model_name", "checkpoint", "group"], how="left")
    return core_agg, group_agg



def _metric_with_se(frame: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    mean = frame[metric].to_numpy(dtype=float)
    se_col = f"{metric}_se"
    se = frame[se_col].to_numpy(dtype=float) if se_col in frame.columns else np.zeros_like(mean)
    return mean, se



def make_overall_vs_worst_group_scatter(model_summary_df: pd.DataFrame, outpath: Path) -> None:
    if model_summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    x = model_summary_df["overall_log_loss"].to_numpy(dtype=float)
    y = model_summary_df["core_worst_group_log_loss"].to_numpy(dtype=float)
    ax.scatter(x, y)
    for _, row in model_summary_df.iterrows():
        ax.annotate(str(row["model_name"]), (float(row["overall_log_loss"]), float(row["core_worst_group_log_loss"])))
    ax.set_xlabel("Overall log-loss")
    ax.set_ylabel("Worst-group log-loss")
    ax.set_title("Overall vs worst-group risk")
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def make_group_risk_bars(group_summary_df: pd.DataFrame, *, baseline_model: str, candidate_model: str, outpath: Path) -> None:
    sub = group_summary_df.loc[group_summary_df["model_name"].isin([baseline_model, candidate_model])].copy()
    if sub.empty:
        return
    groups = sorted(sub["group"].astype(str).unique().tolist())
    x = np.arange(len(groups), dtype=float)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for offset, model_name in [(-width / 2.0, baseline_model), (width / 2.0, candidate_model)]:
        mdf = sub.loc[sub["model_name"] == model_name].set_index("group").reindex(groups)
        mean, se = _metric_with_se(mdf.reset_index(), "log_loss")
        ax.bar(x + offset, mean, width=width, yerr=se, label=model_name, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel("Group log-loss")
    ax.set_title(f"Group risk: {candidate_model} vs {baseline_model}")
    ax.legend()
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def make_group_risk_trajectory_plot(
    trajectory_group_agg: pd.DataFrame,
    *,
    baseline_model: str,
    candidate_model: str,
    outpath: Path,
) -> None:
    sub = trajectory_group_agg.loc[trajectory_group_agg["model_name"].isin([baseline_model, candidate_model])].copy()
    if sub.empty:
        return
    groups = sorted(sub["group"].astype(str).unique().tolist())
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for model_name, linestyle in [(baseline_model, "--"), (candidate_model, "-")]:
        for group_name in groups:
            gdf = sub.loc[(sub["model_name"] == model_name) & (sub["group"] == group_name)].sort_values("checkpoint")
            if gdf.empty:
                continue
            ax.plot(gdf["checkpoint"], gdf["log_loss"], linestyle=linestyle, label=f"{model_name}:{group_name}")
    ax.set_xlabel("Ensemble size / iteration")
    ax.set_ylabel("Group log-loss")
    ax.set_title(f"Group risk trajectory: {candidate_model} vs {baseline_model}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def make_worst_group_trajectory_plot(
    trajectory_core_agg: pd.DataFrame,
    *,
    baseline_model: str,
    candidate_model: str,
    outpath: Path,
) -> None:
    sub = trajectory_core_agg.loc[trajectory_core_agg["model_name"].isin([baseline_model, candidate_model])].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for model_name in [baseline_model, candidate_model]:
        mdf = sub.loc[sub["model_name"] == model_name].sort_values("checkpoint")
        if mdf.empty:
            continue
        ax.plot(mdf["checkpoint"], mdf["core_worst_group_log_loss"], label=model_name)
    ax.set_xlabel("Ensemble size / iteration")
    ax.set_ylabel("Worst-group log-loss")
    ax.set_title(f"Worst-group trajectory: {candidate_model} vs {baseline_model}")
    ax.legend()
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def make_delta_loss_distribution_plot(merged_delta_df: pd.DataFrame, *, baseline_model: str, candidate_model: str, outpath: Path) -> None:
    if merged_delta_df.empty:
        return
    groups = sorted(merged_delta_df["group"].astype(str).unique().tolist())
    data = [merged_delta_df.loc[merged_delta_df["group"] == group, "delta_log_loss"].to_numpy(dtype=float) for group in groups]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.boxplot(data, labels=groups, vert=True, showfliers=False)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("candidate - baseline log-loss")
    ax.set_title(f"Individual loss redistribution: {candidate_model} vs {baseline_model}")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)



def _bootstrap_metric_from_losses(losses: Array, groups: Array, metric: str) -> float:
    if metric == "overall_log_loss":
        return float(np.mean(losses))
    if metric == "core_worst_group_log_loss":
        group_means = [float(np.mean(losses[groups == group_name])) for group_name in pd.unique(groups)]
        return float(np.max(group_means))
    if metric == "core_tail_log_loss_top_10pct":
        k = max(1, int(math.ceil(0.10 * losses.shape[0])))
        return float(np.mean(np.sort(losses)[::-1][:k]))
    if metric == "core_group_log_loss_variance_weighted":
        uniq = pd.unique(groups)
        means = np.asarray([float(np.mean(losses[groups == group_name])) for group_name in uniq], dtype=float)
        weights = np.asarray([float(np.mean(groups == group_name)) for group_name in uniq], dtype=float)
        center = float(np.average(means, weights=weights))
        return float(np.average((means - center) ** 2, weights=weights))
    raise KeyError(f"Unsupported metric={metric!r}")



def bootstrap_pairwise_metric_differences(
    merged_delta_df: pd.DataFrame,
    *,
    baseline_model: str,
    candidate_model: str,
    n_bootstrap: int = 200,
    random_state: int = 0,
    metric_columns: Sequence[str] = LOSS_METRICS,
) -> pd.DataFrame:
    if merged_delta_df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    per_seed = []
    for (dataset_name, seed), sdf in merged_delta_df.groupby(["dataset_name", "seed"], dropna=False):
        per_seed.append(
            {
                "dataset_name": dataset_name,
                "seed": int(seed),
                "baseline_losses": sdf["log_loss_i_baseline"].to_numpy(dtype=float),
                "candidate_losses": sdf["log_loss_i_candidate"].to_numpy(dtype=float),
                "groups": sdf["group"].astype(str).to_numpy(dtype=object),
            }
        )
    if not per_seed:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for metric in metric_columns:
        boot_values = []
        for _ in range(int(n_bootstrap)):
            seed_diffs = []
            for item in per_seed:
                n = item["baseline_losses"].shape[0]
                idx = rng.integers(0, n, size=n)
                base = item["baseline_losses"][idx]
                cand = item["candidate_losses"][idx]
                groups = item["groups"][idx]
                diff = _bootstrap_metric_from_losses(cand, groups, metric) - _bootstrap_metric_from_losses(base, groups, metric)
                seed_diffs.append(diff)
            boot_values.append(float(np.mean(seed_diffs)))
        boot_arr = np.asarray(boot_values, dtype=float)
        rows.append(
            {
                "baseline_model": baseline_model,
                "candidate_model": candidate_model,
                "metric": metric,
                "n_bootstrap": int(n_bootstrap),
                "delta_mean_bootstrap": float(np.mean(boot_arr)),
                "delta_ci_lower_95": float(np.quantile(boot_arr, 0.025)),
                "delta_ci_upper_95": float(np.quantile(boot_arr, 0.975)),
            }
        )
    return pd.DataFrame(rows)
