from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class HeteroscedasticRegressionSplit:
    X: Array
    y: Array
    f_true: Array
    sigma_true: Array
    sample_id: Array
    metadata: Dict[str, Array]


@dataclass(frozen=True)
class HeteroscedasticRegressionDataset:
    dataset_name: str
    train: HeteroscedasticRegressionSplit
    valid: HeteroscedasticRegressionSplit
    calib: HeteroscedasticRegressionSplit
    test: HeteroscedasticRegressionSplit
    feature_names: List[str]
    metadata: Dict[str, object]


@dataclass(frozen=True)
class TabularRegressionSplit:
    X: Array
    y: Array
    sample_id: Array
    metadata: Dict[str, Array]


@dataclass(frozen=True)
class TabularRegressionDataset:
    dataset_name: str
    train: TabularRegressionSplit
    valid: TabularRegressionSplit
    test: TabularRegressionSplit
    feature_names: List[str]
    metadata: Dict[str, object]


@dataclass(frozen=True)
class GroupedPartialLinearSplit:
    X: Array
    D: Array
    y: Array
    group_id: Array
    sample_id: Array
    g_true: Array
    group_effect_true: Array
    eta_true: Array
    metadata: Dict[str, Array]


@dataclass(frozen=True)
class GroupedPartialLinearDataset:
    dataset_name: str
    train: GroupedPartialLinearSplit
    valid: GroupedPartialLinearSplit
    test: GroupedPartialLinearSplit
    beta_true: float
    feature_names: List[str]
    metadata: Dict[str, object]
