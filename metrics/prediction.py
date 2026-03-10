from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

Array = np.ndarray


def _rmse(y_true: Array, y_pred: Array) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _coverage(y_true: Array, lower: Array, upper: Array) -> float:
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def _avg_length(lower: Array, upper: Array) -> float:
    return float(np.mean(upper - lower))


def _conditional_coverages(
    y_true: Array,
    lower: Array,
    upper: Array,
    sigma_true: Array,
    labels: Tuple[str, ...] = ("low", "mid", "high"),
) -> Dict[str, float]:
    q1, q2 = np.quantile(sigma_true, [1.0 / 3.0, 2.0 / 3.0])
    masks = {
        labels[0]: sigma_true <= q1,
        labels[1]: (sigma_true > q1) & (sigma_true <= q2),
        labels[2]: sigma_true > q2,
    }
    out: Dict[str, float] = {}
    for label, mask in masks.items():
        out[f"conditional_coverage_{label}"] = float(np.mean(((y_true >= lower) & (y_true <= upper))[mask]))
    return out


def evaluate_prediction_uq(
    y_true: Array,
    y_pred: Array,
    lower: Array,
    upper: Array,
    sigma_true: Array,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    sigma_true = np.asarray(sigma_true, dtype=float)

    metrics = {
        "rmse": _rmse(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "marginal_coverage": _coverage(y_true, lower, upper),
        "avg_interval_length": _avg_length(lower, upper),
    }
    metrics.update(_conditional_coverages(y_true, lower, upper, sigma_true))
    return metrics
