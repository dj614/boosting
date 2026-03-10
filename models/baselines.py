from __future__ import annotations

from statistics import NormalDist
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from data.types import GroupedPartialLinearDataset, HeteroscedasticRegressionDataset
from models.base import InferenceModel, PredictionModel

Array = np.ndarray


class RandomForestConformalRegressor(PredictionModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 6,
        min_samples_leaf: int = 5,
        alpha: float = 0.1,
        random_state: int = 0,
    ):
        self.alpha = float(alpha)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self._calibration_residuals: Optional[Array] = None

    def fit(self, train_data: HeteroscedasticRegressionDataset) -> "RandomForestConformalRegressor":
        self.model.fit(train_data.train.X, train_data.train.y)
        calib_pred = self.predict(train_data.calib.X)
        self._calibration_residuals = np.abs(train_data.calib.y - calib_pred).astype(float)
        return self

    def predict(self, X: Array) -> Array:
        return np.asarray(self.model.predict(X), dtype=float)

    def predict_interval(self, X: Array, alpha: Optional[float] = None) -> Tuple[Array, Array]:
        if self._calibration_residuals is None:
            raise RuntimeError("Model must be fitted before calling predict_interval")
        alpha = self.alpha if alpha is None else float(alpha)
        q = _conformal_quantile(self._calibration_residuals, alpha=alpha)
        pred = self.predict(X)
        return pred - q, pred + q


class GroupedPartialLinearBaseline(InferenceModel):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 6,
        min_samples_leaf: int = 5,
        random_state: int = 0,
    ):
        self.mu_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self.d_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + 17,
            n_jobs=-1,
        )
        self._beta_hat: Optional[float] = None
        self._beta_se: Optional[float] = None

    def fit(self, train_data: GroupedPartialLinearDataset) -> "GroupedPartialLinearBaseline":
        split = train_data.train
        self.mu_model.fit(split.X, split.y)
        self.d_model.fit(split.X, split.D)

        y_res = split.y - self.mu_model.predict(split.X)
        d_res = split.D - self.d_model.predict(split.X)
        denom = float(np.dot(d_res, d_res))
        if abs(denom) < 1e-12:
            raise RuntimeError("Residualized treatment has near-zero variance; beta is not identifiable")

        beta_hat = float(np.dot(d_res, y_res) / denom)
        self._beta_hat = beta_hat
        self.mu_model.fit(split.X, split.y - beta_hat * split.D)
        self._beta_se = _cluster_robust_se(
            d_res=d_res,
            residual=y_res - beta_hat * d_res,
            group_id=split.group_id,
        )
        return self

    def estimate_beta(self, test_or_eval_data: Optional[object] = None) -> float:
        if self._beta_hat is None:
            raise RuntimeError("Model must be fitted before calling estimate_beta")
        return float(self._beta_hat)

    def estimate_beta_se(self) -> float:
        if self._beta_se is None:
            raise RuntimeError("Model must be fitted before calling estimate_beta_se")
        return float(self._beta_se)

    def confidence_interval(self, alpha: float) -> Tuple[float, float]:
        if self._beta_hat is None or self._beta_se is None:
            raise RuntimeError("Model must be fitted before calling confidence_interval")
        z = NormalDist().inv_cdf(1.0 - float(alpha) / 2.0)
        return float(self._beta_hat - z * self._beta_se), float(self._beta_hat + z * self._beta_se)

    def predict_mu(self, X: Array) -> Array:
        return np.asarray(self.mu_model.predict(X), dtype=float)


def _conformal_quantile(residuals: Array, alpha: float) -> float:
    residuals = np.sort(np.asarray(residuals, dtype=float))
    n = residuals.shape[0]
    level = int(np.ceil((n + 1) * (1.0 - alpha)))
    index = min(max(level - 1, 0), n - 1)
    return float(residuals[index])


def _cluster_robust_se(d_res: Array, residual: Array, group_id: Array) -> float:
    group_arr = np.asarray(group_id, dtype=int)
    unique_groups = np.unique(group_arr)
    denom = float(np.sum(d_res ** 2))
    if abs(denom) < 1e-12:
        return float("nan")

    score_sum_sq = 0.0
    for gid in unique_groups:
        mask = group_arr == gid
        score_g = float(np.sum(d_res[mask] * residual[mask]))
        score_sum_sq += score_g ** 2

    n = d_res.shape[0]
    g = max(len(unique_groups), 1)
    df_correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - 2, 1))
    var_hat = df_correction * score_sum_sq / (denom ** 2)
    return float(np.sqrt(max(var_hat, 0.0)))
