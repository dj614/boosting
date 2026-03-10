from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

Array = np.ndarray


class PredictionModel(ABC):
    @abstractmethod
    def fit(self, train_data: object) -> "PredictionModel":
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Array) -> Array:
        raise NotImplementedError

    @abstractmethod
    def predict_interval(self, X: Array, alpha: Optional[float] = None) -> Tuple[Array, Array]:
        raise NotImplementedError


class InferenceModel(ABC):
    @abstractmethod
    def fit(self, train_data: object) -> "InferenceModel":
        raise NotImplementedError

    @abstractmethod
    def estimate_beta(self, test_or_eval_data: Optional[object] = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def estimate_beta_se(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def confidence_interval(self, alpha: float) -> Tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def predict_mu(self, X: Array) -> Array:
        raise NotImplementedError
