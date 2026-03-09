from __future__ import annotations

from typing import Dict, List, Literal

import numpy as np
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

Array = np.ndarray
TaskType = Literal["regression", "classification"]


def regression_metrics(y_true: Array, pred: Array) -> Dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, pred)),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def classification_metrics(y_true: Array, prob: Array) -> Dict[str, float]:
    prob = np.clip(prob, 1e-8, 1.0 - 1e-8)
    pred_label = (prob >= 0.5).astype(int)
    out = {
        "error_rate": float(np.mean(pred_label != y_true)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["auc"] = float("nan")
    return out


def compute_metrics(task_type: TaskType, y_true: Array, pred: Array) -> Dict[str, float]:
    if task_type == "regression":
        return regression_metrics(y_true, pred)
    return classification_metrics(y_true, pred)


def subgroup_metric_name(task_type: TaskType) -> str:
    return "mse" if task_type == "regression" else "error_rate"


def subgroup_metrics(task_type: TaskType, y_true: Array, pred: Array, meta: Dict[str, Array]) -> Dict[str, float]:
    metric_name = subgroup_metric_name(task_type)
    subgroup_keys: List[str] = [
        key
        for key, value in meta.items()
        if value.ndim == 1 and set(np.unique(value)).issubset({0, 1}) and key != "high_noise"
    ]
    subgroup_keys = ["high_noise"] + sorted(subgroup_keys)

    metrics: Dict[str, float] = {}
    for key in subgroup_keys:
        mask = meta[key].astype(bool)
        if mask.sum() == 0:
            metrics[f"group_{key}_{metric_name}"] = float("nan")
            continue
        sub = compute_metrics(task_type, y_true[mask], pred[mask])
        metrics[f"group_{key}_{metric_name}"] = float(sub[metric_name])
        metrics[f"group_{key}_n"] = float(mask.sum())
    return metrics


def aggregate_prediction_variance(predictions: Array) -> Dict[str, float]:
    pointwise_var = np.var(predictions, axis=0, ddof=0)
    return {
        "pred_var_mean": float(np.mean(pointwise_var)),
        "pred_var_median": float(np.median(pointwise_var)),
        "pred_var_p90": float(np.quantile(pointwise_var, 0.9)),
    }


def groupwise_prediction_variance(predictions: Array, meta: Dict[str, Array]) -> Dict[str, float]:
    pointwise_var = np.var(predictions, axis=0, ddof=0)
    out: Dict[str, float] = {}
    for key, value in meta.items():
        if value.ndim != 1:
            continue
        uniq = np.unique(value)
        if not set(uniq).issubset({0, 1}):
            continue
        mask = value.astype(bool)
        if mask.sum() == 0:
            out[f"pred_var_{key}_mean"] = float("nan")
        else:
            out[f"pred_var_{key}_mean"] = float(np.mean(pointwise_var[mask]))
    return out
