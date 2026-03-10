from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

Array = np.ndarray


def evaluate_grouped_inference(
    *,
    beta_hat: float,
    beta_true: float,
    beta_se: float,
    ci: Tuple[float, float],
    y_true: Array,
    y_pred: Array,
) -> Dict[str, float]:
    lower, upper = ci
    bias = float(beta_hat - beta_true)
    return {
        "beta_bias": bias,
        "beta_rmse": float(abs(bias)),
        "avg_beta_se": float(beta_se),
        "ci_coverage": float(lower <= beta_true <= upper),
        "ci_avg_length": float(upper - lower),
        "predictive_mse": float(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)),
    }
