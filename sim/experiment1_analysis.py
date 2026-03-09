from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _method_order(df: pd.DataFrame) -> List[str]:
    return sorted(df["method"].dropna().astype(str).unique().tolist())


def _primary_metric_direction(task_type: str, primary_metric: str) -> str:
    if task_type == "regression":
        return "min"
    if primary_metric in {"error_rate", "log_loss"}:
        return "min"
    return "max"


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def save_json(payload: Dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_safe_float))


def save_table(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _summarize_methods(trial_df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for method, sub_df in trial_df.groupby("method", dropna=False):
        row: Dict[str, float] = {"method": str(method)}
        numeric_cols = sub_df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            row[f"{col}_mean"] = float(sub_df[col].mean())
            row[f"{col}_se"] = float(sub_df[col].sem())
        row["n_trials"] = float(len(sub_df))
        row["rank_key"] = row.get(f"{primary_metric}_mean", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _subgroup_metric_columns(trial_df: pd.DataFrame, primary_metric: str) -> List[str]:
    suffix = f"_{primary_metric}"
    return [col for col in trial_df.columns if col.startswith("group_") and col.endswith(suffix)]


def build_analysis_summary(trial_df: pd.DataFrame, task_type: str, primary_metric: str) -> Dict[str, object]:
    method_summary_df = _summarize_methods(trial_df, primary_metric=primary_metric)
    direction = _primary_metric_direction(task_type, primary_metric)
    if direction == "min":
        best_idx = method_summary_df[f"{primary_metric}_mean"].astype(float).idxmin()
    else:
        best_idx = method_summary_df[f"{primary_metric}_mean"].astype(float).idxmax()
    best_method = str(method_summary_df.loc[best_idx, "method"])

    subgroup_rows: List[Dict[str, float]] = []
    subgroup_cols = _subgroup_metric_columns(trial_df, primary_metric)
    for method, sub_df in trial_df.groupby("method", dropna=False):
        for col in subgroup_cols:
            subgroup_rows.append(
                {
                    "method": str(method),
                    "group_metric": col,
                    "mean": float(sub_df[col].mean()),
                    "se": float(sub_df[col].sem()),
                }
            )

    return {
        "task_type": task_type,
        "primary_metric": primary_metric,
        "best_method": best_method,
        "method_summary": method_summary_df.sort_values(by=f"{primary_metric}_mean", ascending=(direction == "min")).to_dict("records"),
        "subgroup_summary": subgroup_rows,
    }


def make_pairwise_comparison_table(
    trial_df: pd.DataFrame,
    primary_metric: str,
    baseline_method: str | None = None,
) -> pd.DataFrame:
    methods = _method_order(trial_df)
    if not methods:
        return pd.DataFrame(columns=["baseline_method", "comparison_method", "n_pairs", "mean_delta", "se_delta"])
    baseline = baseline_method or methods[0]
    if baseline not in methods:
        raise ValueError(f"Unknown baseline_method={baseline!r}. Available methods: {methods}")

    pivot = trial_df.pivot_table(index="rep", columns="method", values=primary_metric, aggfunc="mean")
    rows: List[Dict[str, float]] = []
    for method in methods:
        if method == baseline or method not in pivot.columns or baseline not in pivot.columns:
            continue
        diff = pivot[method] - pivot[baseline]
        diff = diff.dropna()
        rows.append(
            {
                "baseline_method": baseline,
                "comparison_method": method,
                "n_pairs": float(diff.shape[0]),
                "mean_delta": float(diff.mean()),
                "se_delta": float(diff.sem()),
            }
        )
    return pd.DataFrame(rows)


def make_method_comparison_plot(
    trial_df: pd.DataFrame,
    task_type: str,
    primary_metric: str,
    outpath: Path,
) -> None:
    methods = _method_order(trial_df)
    if not methods:
        return

    pred_var_col = "pred_var_mean"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metric_ax, var_ax = axes

    metric_values = [trial_df.loc[trial_df["method"] == method, primary_metric].dropna().values for method in methods]
    var_values = [trial_df.loc[trial_df["method"] == method, pred_var_col].dropna().values for method in methods]

    metric_ax.boxplot(metric_values, labels=methods, vert=True)
    metric_ax.set_title(f"Primary metric: {primary_metric}")
    metric_ax.set_ylabel(primary_metric)
    metric_ax.tick_params(axis="x", rotation=35)

    var_ax.boxplot(var_values, labels=methods, vert=True)
    var_ax.set_title("Prediction variance")
    var_ax.set_ylabel(pred_var_col)
    var_ax.tick_params(axis="x", rotation=35)

    fig.suptitle(f"Experiment 1 comparison ({task_type})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def make_error_variance_scatter(
    trial_df: pd.DataFrame,
    task_type: str,
    primary_metric: str,
    outpath: Path,
) -> None:
    pred_var_col = "pred_var_mean"
    grouped = trial_df.groupby("method", dropna=False)[[primary_metric, pred_var_col]].mean().reset_index()
    if grouped.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(grouped[pred_var_col], grouped[primary_metric])
    for _, row in grouped.iterrows():
        ax.annotate(str(row["method"]), (row[pred_var_col], row[primary_metric]))
    ax.set_xlabel(pred_var_col)
    ax.set_ylabel(primary_metric)
    ax.set_title(f"Error-variance tradeoff ({task_type})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _mean_pointwise_prediction(pointwise_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    group_cols = [
        col
        for col in pointwise_df.columns
        if col not in {pred_col, "bootstrap_id"}
    ]
    agg = pointwise_df.groupby(group_cols, dropna=False)[pred_col].mean().reset_index()
    return agg


def _variance_pointwise_prediction(pointwise_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    group_cols = [
        col
        for col in pointwise_df.columns
        if col not in {pred_col, "bootstrap_id"}
    ]
    agg = pointwise_df.groupby(group_cols, dropna=False)[pred_col].var(ddof=0).reset_index(name="pred_var")
    return agg


def _pivot_heatmap(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pivot = df.pivot_table(index="x3_bin", columns="x2_bin", values=value_col, aggfunc="mean")
    x_vals = pivot.columns.to_numpy(dtype=float)
    y_vals = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    return x_vals, y_vals, z


def _render_heatmap(df: pd.DataFrame, value_col: str, title: str, outpath: Path) -> None:
    x_vals, y_vals, z = _pivot_heatmap(df, value_col=value_col)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        z,
        origin="lower",
        aspect="auto",
        extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
    )
    ax.set_xlabel("X[:, 1] bin center")
    ax.set_ylabel("X[:, 2] bin center")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def _subset_slice(pointwise_df: pd.DataFrame) -> pd.DataFrame:
    required = {"method", "rep", "test_index", "f_true", "X_1", "X_2"}
    if not required.issubset(set(pointwise_df.columns)):
        return pd.DataFrame()
    return pointwise_df.copy()


def _maybe_attach_coordinates(pointwise_df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    if "feature_1" in pointwise_df.columns:
        rename_map["feature_1"] = "X_1"
    if "feature_2" in pointwise_df.columns:
        rename_map["feature_2"] = "X_2"
    if rename_map:
        pointwise_df = pointwise_df.rename(columns=rename_map)
    return pointwise_df


def make_slice_heatmaps(
    pointwise_df: pd.DataFrame,
    task_type: str,
    pred_col: str,
    outdir: Path,
    n_bins: int = 30,
) -> None:
    df = _maybe_attach_coordinates(pointwise_df.copy())
    if not {"X_1", "X_2", pred_col, "method"}.issubset(df.columns):
        return

    df = df.copy()
    df["x2_bin"] = pd.cut(df["X_1"], bins=n_bins, labels=False, duplicates="drop")
    df["x3_bin"] = pd.cut(df["X_2"], bins=n_bins, labels=False, duplicates="drop")
    df = df.dropna(subset=["x2_bin", "x3_bin"])
    df["x2_bin"] = df["x2_bin"].astype(int)
    df["x3_bin"] = df["x3_bin"].astype(int)

    x1_centers = df.groupby("x2_bin")["X_1"].mean().rename("x1_center")
    x2_centers = df.groupby("x3_bin")["X_2"].mean().rename("x2_center")
    df = df.merge(x1_centers, left_on="x2_bin", right_index=True, how="left")
    df = df.merge(x2_centers, left_on="x3_bin", right_index=True, how="left")
    df = df.rename(columns={"x1_center": "x2_bin_center", "x2_center": "x3_bin_center"})

    mean_df = _mean_pointwise_prediction(df, pred_col=pred_col)
    var_df = _variance_pointwise_prediction(df, pred_col=pred_col)

    for method in _method_order(df):
        mean_sub = mean_df.loc[mean_df["method"] == method].copy()
        var_sub = var_df.loc[var_df["method"] == method].copy()
        if mean_sub.empty or var_sub.empty:
            continue
        _render_heatmap(
            mean_sub,
            value_col=pred_col,
            title=f"Mean prediction surface: {method} ({task_type})",
            outpath=outdir / f"{method}_mean_surface.png",
        )
        _render_heatmap(
            var_sub,
            value_col="pred_var",
            title=f"Prediction variance surface: {method} ({task_type})",
            outpath=outdir / f"{method}_variance_surface.png",
        )


def add_feature_columns_to_pointwise(pointwise_df: pd.DataFrame, X_test: np.ndarray) -> pd.DataFrame:
    df = pointwise_df.copy()
    if "test_index" not in df.columns:
        return df
    for idx in range(min(3, X_test.shape[1])):
        df[f"feature_{idx}"] = X_test[df["test_index"].astype(int).to_numpy(), idx]
    return df
