from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score


Array = np.ndarray



def _clip_prob(y_prob: Sequence[float]) -> Array:
    return np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1.0 - 1e-8)



def binary_log_loss_per_sample(y_true: Sequence[int], y_prob: Sequence[float]) -> Array:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = _clip_prob(y_prob)
    return -(y_true_arr * np.log(y_prob_arr) + (1 - y_true_arr) * np.log(1.0 - y_prob_arr))



def binary_brier_per_sample(y_true: Sequence[int], y_prob: Sequence[float]) -> Array:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = _clip_prob(y_prob)
    return (y_prob_arr - y_true_arr) ** 2



def binary_margin(y_prob: Sequence[float]) -> Array:
    return np.abs(_clip_prob(y_prob) - 0.5)



def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = _clip_prob(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=False)

    total = y_true_arr.shape[0]
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true_arr[mask]))
        conf = float(np.mean(y_prob_arr[mask]))
        ece += (mask.sum() / total) * abs(acc - conf)
    return float(ece)



def compute_binary_classification_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = _clip_prob(y_prob)
    y_pred = (y_prob_arr >= threshold).astype(int)

    out = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "log_loss": float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1])),
        "brier": float(np.mean(binary_brier_per_sample(y_true_arr, y_prob_arr))),
        "calibration_error": float(expected_calibration_error(y_true_arr, y_prob_arr)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out



def compute_groupwise_binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    group: Sequence[object],
    threshold: float = 0.5,
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = _clip_prob(y_prob)
    group_arr = np.asarray(group, dtype=object)
    rows: List[Dict[str, float]] = []

    for group_name in sorted(pd.unique(group_arr).tolist()):
        mask = group_arr == group_name
        metrics = compute_binary_classification_metrics(y_true_arr[mask], y_prob_arr[mask], threshold=threshold)
        rows.append(
            {
                "group": str(group_name),
                "n": int(mask.sum()),
                "positive_rate": float(np.mean(y_true_arr[mask])),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)



def compute_risk_redistribution_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    group: Sequence[object],
    tail_fracs: Tuple[float, ...] = (0.05, 0.10),
) -> Dict[str, float]:
    losses = binary_log_loss_per_sample(y_true, y_prob)
    group_df = compute_groupwise_binary_metrics(y_true=y_true, y_prob=y_prob, group=group)
    group_risks = group_df["log_loss"].to_numpy(dtype=float)
    group_weights = group_df["n"].to_numpy(dtype=float)
    group_weights = group_weights / group_weights.sum()

    out = {
        "worst_group_log_loss": float(np.max(group_risks)),
        "group_log_loss_variance": float(np.var(group_risks, ddof=0)),
        "group_log_loss_variance_weighted": float(np.average((group_risks - np.average(group_risks, weights=group_weights)) ** 2, weights=group_weights)),
        "group_log_loss_gap": float(np.max(group_risks) - np.min(group_risks)),
    }

    sorted_losses = np.sort(losses)[::-1]
    n = losses.shape[0]
    for frac in tail_fracs:
        k = max(1, int(np.ceil(frac * n)))
        pct = int(round(100 * frac))
        out[f"tail_log_loss_top_{pct}pct"] = float(np.mean(sorted_losses[:k]))
    return out



def hard_group_gain_vs_easy_group_sacrifice(
    baseline_prob: Sequence[float],
    candidate_prob: Sequence[float],
    y_true: Sequence[int],
    group: Sequence[object],
    hard_groups: Iterable[object],
    easy_groups: Iterable[object],
    lambda_easy: float = 1.0,
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    group_arr = np.asarray(group, dtype=object)
    baseline_losses = binary_log_loss_per_sample(y_true_arr, baseline_prob)
    candidate_losses = binary_log_loss_per_sample(y_true_arr, candidate_prob)
    delta = baseline_losses - candidate_losses

    hard_mask = np.isin(group_arr, list(hard_groups))
    easy_mask = np.isin(group_arr, list(easy_groups))
    if not np.any(hard_mask):
        raise ValueError("No samples matched hard_groups")
    if not np.any(easy_mask):
        raise ValueError("No samples matched easy_groups")

    hard_gain = float(np.mean(delta[hard_mask]))
    easy_gain = float(np.mean(delta[easy_mask]))
    easy_sacrifice = -easy_gain
    return {
        "hard_group_gain": hard_gain,
        "easy_group_gain": easy_gain,
        "easy_group_sacrifice": easy_sacrifice,
        "hard_minus_lambda_easy": float(hard_gain - lambda_easy * easy_sacrifice),
    }



def make_binary_prediction_frame(
    *,
    sample_id: Sequence[object],
    dataset_name: str,
    split: str,
    seed: int,
    model_name: str,
    group: Sequence[object],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    metadata: Optional[Mapping[str, Sequence[object]]] = None,
) -> pd.DataFrame:
    sample_id_arr = np.asarray(sample_id, dtype=object)
    group_arr = np.asarray(group, dtype=object)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = _clip_prob(y_prob)
    y_pred = (y_prob_arr >= 0.5).astype(int)

    frame = pd.DataFrame(
        {
            "sample_id": sample_id_arr,
            "dataset_name": dataset_name,
            "split": split,
            "seed": int(seed),
            "model_name": model_name,
            "group": group_arr,
            "y_true": y_true_arr,
            "y_prob": y_prob_arr,
            "y_pred": y_pred,
            "log_loss_i": binary_log_loss_per_sample(y_true_arr, y_prob_arr),
            "brier_i": binary_brier_per_sample(y_true_arr, y_prob_arr),
            "correct": (y_pred == y_true_arr).astype(int),
            "margin": binary_margin(y_prob_arr),
        }
    )

    if metadata is not None:
        for key, values in metadata.items():
            arr = np.asarray(values)
            if arr.shape[0] != frame.shape[0]:
                raise ValueError(f"metadata column {key!r} has length {arr.shape[0]}, expected {frame.shape[0]}")
            frame[key] = arr
    return frame



def evaluate_binary_predictions(
    *,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    group: Sequence[object],
    threshold: float = 0.5,
) -> Dict[str, object]:
    overall = compute_binary_classification_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    group_df = compute_groupwise_binary_metrics(y_true=y_true, y_prob=y_prob, group=group, threshold=threshold)
    core = compute_risk_redistribution_metrics(y_true=y_true, y_prob=y_prob, group=group)
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
