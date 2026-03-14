from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from data.grouped_partial_linear import generate_grouped_partial_linear_dataset


Array = np.ndarray


@dataclass(frozen=True)
class GroupedRegressionSplit:
    X: Array
    y: Array
    group: Array
    sample_id: Array
    difficulty_score: Optional[Array]
    metadata: Dict[str, Array]


@dataclass(frozen=True)
class GroupedRegressionDataset:
    dataset_name: str
    train: GroupedRegressionSplit
    valid: GroupedRegressionSplit
    test: GroupedRegressionSplit
    feature_names: List[str]
    group_names: List[str]
    metadata: Dict[str, object]


def _as_group_names(values: Sequence[int]) -> Array:
    return np.asarray([f"group_{int(v)}" for v in values], dtype=object)


def _wrap_split(split) -> GroupedRegressionSplit:
    X = np.column_stack([np.asarray(split.D, dtype=float).reshape(-1, 1), np.asarray(split.X, dtype=float)])
    metadata = dict(split.metadata)
    metadata.update(
        {
            "group_id": np.asarray(split.group_id, dtype=int),
            "treatment": np.asarray(split.D, dtype=float),
            "g_true": np.asarray(split.g_true, dtype=float),
            "group_effect_true": np.asarray(split.group_effect_true, dtype=float),
            "eta_true": np.asarray(split.eta_true, dtype=float),
        }
    )
    return GroupedRegressionSplit(
        X=X,
        y=np.asarray(split.y, dtype=float),
        group=_as_group_names(split.group_id),
        sample_id=np.asarray(split.sample_id, dtype=object),
        difficulty_score=np.abs(np.asarray(split.eta_true, dtype=float)),
        metadata=metadata,
    )


def simulate_grouped_regression(
    *,
    n_samples: int = 12000,
    n_features: int = 8,
    valid_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 0,
    group_size: int = 8,
) -> GroupedRegressionDataset:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    n_groups = max(12, int(np.ceil(float(n_samples) / float(group_size))))
    base = generate_grouped_partial_linear_dataset(
        n_groups=n_groups,
        group_size=int(group_size),
        n_features=max(5, int(n_features) - 1),
        valid_group_frac=float(valid_size),
        test_group_frac=float(test_size),
        seed=int(random_state),
    )
    train = _wrap_split(base.train)
    valid = _wrap_split(base.valid)
    test = _wrap_split(base.test)
    group_names = sorted(pd.unique(np.concatenate([train.group, valid.group, test.group])).tolist())
    metadata = dict(base.metadata)
    metadata.update(
        {
            "task_type": "regression",
            "group_size": int(group_size),
            "n_requested_samples": int(n_samples),
            "beta_true": float(base.beta_true),
        }
    )
    return GroupedRegressionDataset(
        dataset_name=f"grouped_partial_linear_n{int(n_groups * group_size)}",
        train=train,
        valid=valid,
        test=test,
        feature_names=["D", *list(base.feature_names)],
        group_names=group_names,
        metadata=metadata,
    )


def summarize_grouped_regression_dataset(dataset: GroupedRegressionDataset) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "dataset_name": dataset.dataset_name,
        "n_features": int(len(dataset.feature_names)),
        "group_names": list(dataset.group_names),
    }
    for split_name in ["train", "valid", "test"]:
        split = getattr(dataset, split_name)
        summary[f"{split_name}_n"] = int(split.y.shape[0])
        summary[f"{split_name}_y_mean"] = float(np.mean(split.y))
        summary[f"{split_name}_y_std"] = float(np.std(split.y))
        group_counts = pd.Series(split.group).value_counts(dropna=False).sort_index()
        summary[f"{split_name}_group_counts"] = {str(k): int(v) for k, v in group_counts.items()}
        group_mean = (
            pd.DataFrame({"group": split.group, "y": split.y})
            .groupby("group", dropna=False)["y"]
            .mean()
            .sort_index()
        )
        summary[f"{split_name}_group_target_means"] = {str(k): float(v) for k, v in group_mean.items()}
        if split.difficulty_score is not None:
            summary[f"{split_name}_difficulty_score_mean"] = float(np.mean(split.difficulty_score))
    return summary
