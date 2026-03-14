from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


Array = np.ndarray


def squared_error_per_sample(y_true: Sequence[float], y_pred: Sequence[float]) -> Array:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return (y_pred_arr - y_true_arr) ** 2


def absolute_error_per_sample(y_true: Sequence[float], y_pred: Sequence[float]) -> Array:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return np.abs(y_pred_arr - y_true_arr)


def compute_regression_metrics(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


def compute_groupwise_regression_metrics(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    group: Sequence[object],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "group": np.asarray(group, dtype=object),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    rows = []
    for group_name, sub_df in frame.groupby("group", dropna=False):
        metrics = compute_regression_metrics(y_true=sub_df["y_true"].to_numpy(), y_pred=sub_df["y_pred"].to_numpy())
        rows.append(
            {
                "group": group_name,
                "n": int(sub_df.shape[0]),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def compute_risk_redistribution_metrics(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    group: Sequence[object],
    tail_fracs: Sequence[float] = (0.05, 0.10),
) -> Dict[str, float]:
    se = squared_error_per_sample(y_true, y_pred)
    group_df = compute_groupwise_regression_metrics(y_true=y_true, y_pred=y_pred, group=group)
    if group_df.empty:
        raise ValueError("group must contain at least one non-empty group")
    group_risks = group_df["mse"].to_numpy(dtype=float)
    group_weights = group_df["n"].to_numpy(dtype=float)
    out = {
        "worst_group_mse": float(np.max(group_risks)),
        "group_mse_variance": float(np.var(group_risks, ddof=0)),
        "group_mse_variance_weighted": float(
            np.average((group_risks - np.average(group_risks, weights=group_weights)) ** 2, weights=group_weights)
        ),
        "group_mse_gap": float(np.max(group_risks) - np.min(group_risks)),
    }
    sorted_losses = np.sort(se)[::-1]
    n = se.shape[0]
    for frac in tail_fracs:
        k = max(1, int(np.ceil(float(frac) * n)))
        pct = int(round(100 * float(frac)))
        out[f"tail_squared_error_top_{pct}pct"] = float(np.mean(sorted_losses[:k]))
    return out


def make_regression_prediction_frame(
    *,
    sample_id: Sequence[object],
    dataset_name: str,
    split: str,
    seed: int,
    model_name: str,
    family_name: str,
    group: Sequence[object],
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metadata: Optional[Mapping[str, Sequence[object]]] = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": np.asarray(sample_id, dtype=object),
            "dataset_name": dataset_name,
            "split": split,
            "seed": int(seed),
            "model_name": model_name,
            "family_name": family_name,
            "group": np.asarray(group, dtype=object),
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )
    frame["squared_error_i"] = squared_error_per_sample(frame["y_true"], frame["y_pred"])
    frame["absolute_error_i"] = absolute_error_per_sample(frame["y_true"], frame["y_pred"])
    if metadata is not None:
        for key, values in metadata.items():
            arr = np.asarray(values)
            if arr.shape[0] != frame.shape[0]:
                raise ValueError(f"metadata column {key!r} has length {arr.shape[0]}, expected {frame.shape[0]}")
            frame[key] = arr
    return frame


def evaluate_regression_predictions(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    group: Sequence[object],
) -> Dict[str, object]:
    overall = compute_regression_metrics(y_true=y_true, y_pred=y_pred)
    group_df = compute_groupwise_regression_metrics(y_true=y_true, y_pred=y_pred, group=group)
    core = compute_risk_redistribution_metrics(y_true=y_true, y_pred=y_pred, group=group)
    return {
        "overall": overall,
        "group_metrics": group_df,
        "core_risk": core,
    }


def save_prediction_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("Expected output suffix .csv or .parquet")
