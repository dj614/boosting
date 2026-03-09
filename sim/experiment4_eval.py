from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


Array = np.ndarray


def regression_metrics(y_true: Array, y_pred: Array) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    residual = y_true - y_pred
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    y_centered = y_true - float(np.mean(y_true))
    denom = float(np.sum(y_centered ** 2))
    r2 = float(1.0 - np.sum(residual ** 2) / denom) if denom > 0.0 else float("nan")
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": r2,
    }


def support_indicator(support: Sequence[int], p: int) -> Array:
    if p <= 0:
        raise ValueError("p must be positive")
    indicator = np.zeros(int(p), dtype=int)
    if support is None:
        return indicator
    idx = np.asarray(list(support), dtype=int).reshape(-1)
    if idx.size == 0:
        return indicator
    if np.any(idx < 0) or np.any(idx >= p):
        raise ValueError("support indices must lie in [0, p)")
    indicator[np.unique(idx)] = 1
    return indicator


def support_recovery_metrics(
    support_true: Sequence[int],
    support_hat: Sequence[int],
    p: Optional[int] = None,
) -> Dict[str, float]:
    true_idx = np.unique(np.asarray(list(support_true), dtype=int).reshape(-1))
    hat_idx = np.unique(np.asarray(list(support_hat), dtype=int).reshape(-1))

    if p is None:
        max_idx = -1
        if true_idx.size:
            max_idx = max(max_idx, int(true_idx.max()))
        if hat_idx.size:
            max_idx = max(max_idx, int(hat_idx.max()))
        p = max_idx + 1 if max_idx >= 0 else 1

    true_mask = support_indicator(true_idx, p=int(p))
    hat_mask = support_indicator(hat_idx, p=int(p))

    tp = float(np.sum((true_mask == 1) & (hat_mask == 1)))
    fp = float(np.sum((true_mask == 0) & (hat_mask == 1)))
    fn = float(np.sum((true_mask == 1) & (hat_mask == 0)))
    tn = float(np.sum((true_mask == 0) & (hat_mask == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    union = float(np.sum((true_mask == 1) | (hat_mask == 1)))
    jaccard = float(tp / union) if union > 0 else 1.0

    return {
        "support_size_true": float(true_idx.size),
        "support_size_hat": float(hat_idx.size),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "exact_support_recovery": float(np.array_equal(true_mask, hat_mask)),
    }


def stability_selection_metrics(supports: Sequence[Sequence[int]], p: int) -> Dict[str, object]:
    support_list = [np.unique(np.asarray(list(s), dtype=int).reshape(-1)) for s in supports]
    if p <= 0:
        raise ValueError("p must be positive")
    if not support_list:
        return {
            "n_supports": 0.0,
            "mean_support_size": float("nan"),
            "selection_frequency": np.zeros(int(p), dtype=float),
            "pairwise_jaccard_mean": float("nan"),
            "pairwise_jaccard_se": float("nan"),
        }

    indicator_matrix = np.vstack([support_indicator(s, p=int(p)) for s in support_list]).astype(float)
    selection_frequency = indicator_matrix.mean(axis=0)
    support_sizes = indicator_matrix.sum(axis=1)

    jaccards: List[float] = []
    for i in range(len(support_list)):
        for j in range(i + 1, len(support_list)):
            a = indicator_matrix[i] > 0.5
            b = indicator_matrix[j] > 0.5
            union = float(np.sum(a | b))
            inter = float(np.sum(a & b))
            jaccards.append(inter / union if union > 0 else 1.0)

    jaccard_arr = np.asarray(jaccards, dtype=float)
    return {
        "n_supports": float(len(support_list)),
        "mean_support_size": float(np.mean(support_sizes)),
        "support_size_se": float(pd.Series(support_sizes).sem()) if len(support_sizes) > 1 else float("nan"),
        "selection_frequency": selection_frequency,
        "pairwise_jaccard_mean": float(np.mean(jaccard_arr)) if jaccard_arr.size else float("nan"),
        "pairwise_jaccard_se": float(pd.Series(jaccard_arr).sem()) if jaccard_arr.size > 1 else float("nan"),
    }


def make_feature_support_frame(
    *,
    p: Optional[int] = None,
    supports: Optional[Sequence[Sequence[int]]] = None,
    selection_frequency: Optional[Sequence[float]] = None,
    support_true: Optional[Sequence[int]] = None,
    feature_names: Optional[Sequence[str]] = None,
    extra_columns: Optional[Mapping[str, Sequence[object]]] = None,
) -> pd.DataFrame:
    if selection_frequency is None:
        if supports is None or p is None:
            raise ValueError("Either selection_frequency or both supports and p must be provided")
        stability = stability_selection_metrics(supports=supports, p=int(p))
        freq = np.asarray(stability["selection_frequency"], dtype=float)
        p = int(freq.shape[0])
    else:
        freq = np.asarray(selection_frequency, dtype=float).reshape(-1)
        p = int(freq.shape[0]) if p is None else int(p)
        if freq.shape[0] != p:
            raise ValueError("selection_frequency length must equal p")

    if feature_names is None:
        feature_names = [f"x{idx}" for idx in range(p)]
    if len(feature_names) != p:
        raise ValueError("feature_names length must equal p")

    frame = pd.DataFrame(
        {
            "feature_idx": np.arange(p, dtype=int),
            "feature_name": list(feature_names),
            "selection_frequency": freq.astype(float),
        }
    )
    if support_true is not None:
        frame["is_true_support"] = support_indicator(support_true, p=p).astype(int)
    if extra_columns is not None:
        for key, value in extra_columns.items():
            values = list(value)
            if len(values) != p:
                raise ValueError(f"extra column {key!r} must have length p")
            frame[str(key)] = values
    return frame


def aggregate_metric_table(
    metric_frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    if metric_frame.empty:
        return pd.DataFrame(columns=list(group_cols))

    missing = [col for col in group_cols if col not in metric_frame.columns]
    if missing:
        raise ValueError(f"group_cols missing from metric_frame: {missing}")

    numeric_cols = [
        col for col in metric_frame.select_dtypes(include=[np.number]).columns.tolist() if col not in group_cols
    ]
    rows: List[Dict[str, object]] = []
    for group_key, sub_df in metric_frame.groupby(list(group_cols), dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: Dict[str, object] = {col: value for col, value in zip(group_cols, group_key)}
        row["n_rep"] = float(len(sub_df))
        for col in numeric_cols:
            values = pd.to_numeric(sub_df[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_se"] = float(values.sem())
        rows.append(row)

    out = pd.DataFrame(rows)
    if sort_by is not None and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return out
