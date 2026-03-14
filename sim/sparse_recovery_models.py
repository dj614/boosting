from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error

from .ctb_core import ConsensusTransportBoosting
from .ctb_semantics import ctb_tree_model_name
from .sparse_recovery_data import SparseRegressionSplit

try:  # pragma: no cover
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


Array = np.ndarray


def _sigmoid_probability(score: Array) -> Array:
    score = np.asarray(score, dtype=float).reshape(-1)
    out = np.empty_like(score, dtype=float)
    pos = score >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-score[pos]))
    exp_score = np.exp(score[~pos])
    out[~pos] = exp_score / (1.0 + exp_score)
    return np.clip(out, 1e-8, 1.0 - 1e-8)


def _binary_log_loss(y_true: Array, y_prob: Array) -> float:
    return float(log_loss(np.asarray(y_true, dtype=int), np.clip(np.asarray(y_prob, dtype=float), 1e-8, 1.0 - 1e-8), labels=[0, 1]))


def _binary_accuracy(y_true: Array, y_prob: Array) -> float:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    return float(accuracy_score(y_true_arr, (y_prob_arr >= 0.5).astype(int)))


@dataclass(frozen=True)
class L2BoostingConfig:
    max_steps: int = 300
    learning_rate: float = 0.1
    coef_tol: float = 1e-10
    random_state: int = 0

    @property
    def model_name(self) -> str:
        return "l2boost"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaggedComponentwiseConfig:
    n_estimators: int = 100
    base_max_steps: int = 100
    learning_rate: float = 0.1
    bootstrap_fraction: float = 1.0
    support_frequency_threshold: float = 0.5
    coef_tol: float = 1e-10
    random_state: int = 0

    @property
    def model_name(self) -> str:
        return "bagged_componentwise"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class CTBSparseConfig:
    max_steps: int = 300
    n_inner_bootstraps: int = 8
    eta: float = 1.0
    enable_group_consensus: bool = False
    group_corr_threshold: float = 0.9
    residual_weight_power: float = 1.0
    residual_weight_eps: float = 1e-8
    consensus_frequency_power: float = 2.0
    consensus_sign_power: float = 1.0
    instability_lambda: float = 1.0
    instability_power: float = 1.0
    min_consensus_frequency: float = 0.25
    min_sign_consistency: float = 0.75
    support_frequency_threshold: float = 0.05
    coef_tol: float = 1e-10
    random_state: int = 0

    @property
    def model_name(self) -> str:
        return "ctb_sparse"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class LassoPathConfig:
    alphas: Optional[Tuple[float, ...]] = None
    n_alphas: int = 80
    eps: float = 1e-3
    coef_tol: float = 1e-10
    random_state: int = 0

    @property
    def model_name(self) -> str:
        return "lasso"

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        if self.alphas is not None:
            payload["alphas"] = [float(a) for a in self.alphas]
        return payload


@dataclass(frozen=True)
class XGBTreeConfig:
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    random_state: int = 0

    @property
    def model_name(self) -> str:
        return "xgb_tree"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CTBTreeConfig:
    n_estimators: int = 300
    n_inner_bootstraps: int = 8
    eta: float = 1.0
    max_depth: int = 1
    min_samples_leaf: int = 5
    instability_penalty: float = 0.0
    weight_power: float = 1.0
    weight_eps: float = 1e-8
    update_target_mode: str = "legacy"
    transport_curvature_eps: float = 1e-6
    random_state: int = 0

    @property
    def model_name(self) -> str:
        return ctb_tree_model_name(
            depth=int(self.max_depth),
            update_target_mode=str(self.update_target_mode),
            transport_curvature_eps=float(self.transport_curvature_eps),
            include_task_suffix=False,
        ).replace("ctb_depth", "ctb_tree_depth", 1)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StandardizationStats:
    x_mean: Array
    x_std: Array
    y_mean: float


@dataclass(frozen=True)
class L2BoostPathResult:
    coef_path: Array
    selected_feature_path: Array
    train_prediction_path: Array


class SparseRegressionWrapperBase:
    def __init__(self) -> None:
        self.standardization_: Optional[StandardizationStats] = None
        self.selected_support_: Optional[Array] = None
        self.selection_trace_: Optional[pd.DataFrame] = None

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def fit(
        self,
        train_split: SparseRegressionSplit,
        valid_split: SparseRegressionSplit,
    ) -> "SparseRegressionWrapperBase":
        raise NotImplementedError

    def predict(self, X: Array):
        raise NotImplementedError


class SparseClassificationWrapperBase(SparseRegressionWrapperBase):
    def predict_proba(self, X: Array, **kwargs) -> Array:
        raise NotImplementedError

    def predict(self, X: Array, threshold: float = 0.5, **kwargs) -> Array:
        return (self.predict_proba(X, **kwargs) >= float(threshold)).astype(int)


class L2BoostingRegressorWrapper(SparseRegressionWrapperBase):
    def __init__(
        self,
        config: L2BoostingConfig,
        selection_checkpoints: Optional[Sequence[int]] = None,
        trajectory_checkpoints: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.selection_checkpoints = _resolve_checkpoints(
            selection_checkpoints,
            max_checkpoint=config.max_steps,
            default_points=min(25, config.max_steps),
        )
        self.trajectory_checkpoints = _resolve_checkpoints(
            trajectory_checkpoints,
            max_checkpoint=config.max_steps,
            default_points=min(25, config.max_steps),
        )
        self.selected_step_: Optional[int] = None
        self.coef_path_: Optional[Array] = None
        self.selected_feature_path_: Optional[Array] = None
        self.entry_step_: Optional[Array] = None
        self.selected_coef_: Optional[Array] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(
        self,
        train_split: SparseRegressionSplit,
        valid_split: SparseRegressionSplit,
    ) -> "L2BoostingRegressorWrapper":
        X_train_std, y_train_centered, stats = _prepare_train_arrays(train_split.X, train_split.y)
        X_valid_std = _transform_features(valid_split.X, stats)
        self.standardization_ = stats

        path = _fit_l2boost_path(
            X=X_train_std,
            y_centered=y_train_centered,
            max_steps=self.config.max_steps,
            learning_rate=self.config.learning_rate,
        )
        self.coef_path_ = path.coef_path
        self.selected_feature_path_ = path.selected_feature_path
        self.entry_step_ = _compute_entry_step(path.selected_feature_path, X_train_std.shape[1])

        valid_pred_path = _predict_centered_from_coef_path(X_valid_std, self.coef_path_)
        rows: List[Dict[str, float]] = []
        best_step = None
        best_valid_mse = None
        for step in self.selection_checkpoints:
            pred = valid_pred_path[step - 1] + stats.y_mean
            mse = float(mean_squared_error(valid_split.y, pred))
            coef = self.coef_path_[step - 1]
            support_size = int(np.count_nonzero(np.abs(coef) > self.config.coef_tol))
            rows.append(
                {
                    "checkpoint": int(step),
                    "valid_mse": mse,
                    "support_size": float(support_size),
                    "last_selected_feature": float(self.selected_feature_path_[step - 1]),
                }
            )
            if best_valid_mse is None or mse < best_valid_mse:
                best_valid_mse = mse
                best_step = int(step)

        self.selection_trace_ = pd.DataFrame(rows)
        self.selected_step_ = int(best_step)
        self.selected_coef_ = self.coef_path_[self.selected_step_ - 1].copy()
        self.selected_support_ = np.flatnonzero(np.abs(self.selected_coef_) > self.config.coef_tol).astype(int)
        return self

    def coef_at_step(self, step: Optional[int] = None) -> Array:
        if self.coef_path_ is None:
            raise RuntimeError("Model has not been fit")
        use_step = int(step or self.selected_step_ or self.config.max_steps)
        _validate_step(use_step, self.config.max_steps)
        return self.coef_path_[use_step - 1].copy()

    def predict(self, X: Array, step: Optional[int] = None) -> Array:
        stats = _require_standardization(self.standardization_)
        coef = self.coef_at_step(step=step)
        X_std = _transform_features(X, stats)
        return X_std @ coef + stats.y_mean

    def predict_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.coef_path_ is None:
            raise RuntimeError("Model has not been fit")
        stats = _require_standardization(self.standardization_)
        X_std = _transform_features(X, stats)
        out: Dict[int, Array] = {}
        for step in sorted({int(s) for s in checkpoints}):
            _validate_step(step, self.config.max_steps)
            out[int(step)] = X_std @ self.coef_path_[step - 1] + stats.y_mean
        return out

    def support_at_step(self, step: Optional[int] = None) -> Array:
        coef = self.coef_at_step(step=step)
        return np.flatnonzero(np.abs(coef) > self.config.coef_tol).astype(int)


class BaggedComponentwiseRegressorWrapper(SparseRegressionWrapperBase):
    def __init__(
        self,
        config: BaggedComponentwiseConfig,
        selection_checkpoints: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.selection_checkpoints = _resolve_checkpoints(
            selection_checkpoints,
            max_checkpoint=config.n_estimators,
            default_points=min(20, config.n_estimators),
        )
        self.selected_checkpoint_: Optional[int] = None
        self.bag_coef_matrix_: Optional[Array] = None
        self.bag_selected_feature_path_: Optional[Array] = None
        self.selected_coef_: Optional[Array] = None
        self.selection_frequency_: Optional[Array] = None
        self.average_abs_coef_: Optional[Array] = None
        self.mean_support_size_by_checkpoint_: Optional[Dict[int, float]] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(
        self,
        train_split: SparseRegressionSplit,
        valid_split: SparseRegressionSplit,
    ) -> "BaggedComponentwiseRegressorWrapper":
        rng = np.random.default_rng(self.config.random_state)
        X_train_std, y_train_centered, stats = _prepare_train_arrays(train_split.X, train_split.y)
        X_valid_std = _transform_features(valid_split.X, stats)
        self.standardization_ = stats

        n_train, p = X_train_std.shape
        bag_size = max(1, int(round(self.config.bootstrap_fraction * n_train)))
        coef_matrix = np.zeros((self.config.n_estimators, p), dtype=float)
        selected_path_matrix = np.zeros((self.config.n_estimators, self.config.base_max_steps), dtype=int)
        valid_preds = np.zeros((self.config.n_estimators, X_valid_std.shape[0]), dtype=float)
        support_sizes = np.zeros(self.config.n_estimators, dtype=float)

        for bag_idx in range(self.config.n_estimators):
            sample_idx = rng.choice(n_train, size=bag_size, replace=True)
            path = _fit_l2boost_path(
                X=X_train_std[sample_idx],
                y_centered=y_train_centered[sample_idx],
                max_steps=self.config.base_max_steps,
                learning_rate=self.config.learning_rate,
            )
            coef = path.coef_path[-1]
            coef_matrix[bag_idx] = coef
            selected_path_matrix[bag_idx] = path.selected_feature_path
            valid_preds[bag_idx] = X_valid_std @ coef + stats.y_mean
            support_sizes[bag_idx] = float(np.count_nonzero(np.abs(coef) > self.config.coef_tol))

        self.bag_coef_matrix_ = coef_matrix
        self.bag_selected_feature_path_ = selected_path_matrix

        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_valid_mse = None
        mean_support_size_by_checkpoint: Dict[int, float] = {}
        for checkpoint in self.selection_checkpoints:
            avg_pred = valid_preds[:checkpoint].mean(axis=0)
            mse = float(mean_squared_error(valid_split.y, avg_pred))
            avg_coef = coef_matrix[:checkpoint].mean(axis=0)
            selection_freq = np.mean(np.abs(coef_matrix[:checkpoint]) > self.config.coef_tol, axis=0)
            threshold_support = int(np.count_nonzero(selection_freq >= self.config.support_frequency_threshold))
            mean_support = float(np.mean(support_sizes[:checkpoint]))
            mean_support_size_by_checkpoint[int(checkpoint)] = mean_support
            rows.append(
                {
                    "checkpoint": int(checkpoint),
                    "valid_mse": mse,
                    "support_size_frequency_threshold": float(threshold_support),
                    "mean_single_model_support_size": mean_support,
                    "union_support_size": float(np.count_nonzero(np.any(np.abs(coef_matrix[:checkpoint]) > self.config.coef_tol, axis=0))),
                }
            )
            if best_valid_mse is None or mse < best_valid_mse:
                best_valid_mse = mse
                best_checkpoint = int(checkpoint)

        self.selection_trace_ = pd.DataFrame(rows)
        self.selected_checkpoint_ = int(best_checkpoint)
        self.selected_coef_ = coef_matrix[: self.selected_checkpoint_].mean(axis=0)
        self.selection_frequency_ = np.mean(
            np.abs(coef_matrix[: self.selected_checkpoint_]) > self.config.coef_tol,
            axis=0,
        )
        self.average_abs_coef_ = np.mean(np.abs(coef_matrix[: self.selected_checkpoint_]), axis=0)
        self.selected_support_ = np.flatnonzero(
            self.selection_frequency_ >= self.config.support_frequency_threshold
        ).astype(int)
        self.mean_support_size_by_checkpoint_ = mean_support_size_by_checkpoint
        return self

    def predict(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        out = self.predict_staged(X, checkpoints=[int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)])
        key = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        return out[key]

    def predict_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.bag_coef_matrix_ is None:
            raise RuntimeError("Model has not been fit")
        stats = _require_standardization(self.standardization_)
        X_std = _transform_features(X, stats)
        requested = sorted({int(c) for c in checkpoints})
        out: Dict[int, Array] = {}
        for checkpoint in requested:
            _validate_step(checkpoint, self.config.n_estimators)
            coef = self.bag_coef_matrix_[:checkpoint].mean(axis=0)
            out[int(checkpoint)] = X_std @ coef + stats.y_mean
        return out

    def selection_frequency_at_checkpoint(self, checkpoint: Optional[int] = None) -> Array:
        if self.bag_coef_matrix_ is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        _validate_step(use_checkpoint, self.config.n_estimators)
        return np.mean(np.abs(self.bag_coef_matrix_[:use_checkpoint]) > self.config.coef_tol, axis=0)

    def support_at_checkpoint(self, checkpoint: Optional[int] = None, threshold: Optional[float] = None) -> Array:
        freq = self.selection_frequency_at_checkpoint(checkpoint=checkpoint)
        tau = float(self.config.support_frequency_threshold if threshold is None else threshold)
        return np.flatnonzero(freq >= tau).astype(int)

class CTBSparseRegressorWrapper(SparseRegressionWrapperBase):
    def __init__(
        self,
        config: CTBSparseConfig,
        selection_checkpoints: Optional[Sequence[int]] = None,
        trajectory_checkpoints: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.selection_checkpoints = _resolve_checkpoints(
            selection_checkpoints,
            max_checkpoint=config.max_steps,
            default_points=min(25, config.max_steps),
        )
        self.trajectory_checkpoints = _resolve_checkpoints(
            trajectory_checkpoints,
            max_checkpoint=config.max_steps,
            default_points=min(25, config.max_steps),
        )
        self.selected_step_: Optional[int] = None
        self.coef_path_: Optional[Array] = None
        self.step_update_l1_path_: Optional[Array] = None
        self.step_size_path_: Optional[Array] = None
        self.mean_instability_path_: Optional[Array] = None
        self.group_support_score_path_: Optional[Array] = None
        self.feature_groups_: Optional[List[Array]] = None
        self.feature_to_group_: Optional[Array] = None
        self.support_score_path_: Optional[Array] = None
        self.consensus_weight_path_: Optional[Array] = None
        self.conditional_mean_gamma_path_: Optional[Array] = None
        self.sign_consistency_path_: Optional[Array] = None
        self.selected_feature_matrix_: Optional[Array] = None
        self.selection_frequency_path_: Optional[Array] = None
        self.selection_frequency_: Optional[Array] = None
        self.support_score_: Optional[Array] = None
        self.selected_coef_: Optional[Array] = None
        self.selected_support_from_frequency_: Optional[Array] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(
        self,
        train_split: SparseRegressionSplit,
        valid_split: SparseRegressionSplit,
    ) -> "CTBSparseRegressorWrapper":
        X_train_std, y_train_centered, stats = _prepare_train_arrays(train_split.X, train_split.y)
        X_valid_std = _transform_features(valid_split.X, stats)
        self.standardization_ = stats

        n_train, p = X_train_std.shape
        if self.config.enable_group_consensus:
            feature_groups, feature_to_group = _build_correlation_groups(
                X_train_std,
                self.config.group_corr_threshold,
            )
        else:
            feature_groups = [np.array([j], dtype=int) for j in range(p)]
            feature_to_group = np.arange(p, dtype=int)
        n_groups = len(feature_groups)
        self.feature_groups_ = feature_groups
        self.feature_to_group_ = feature_to_group
        rng = np.random.default_rng(self.config.random_state)
        coef = np.zeros(p, dtype=float)
        pred = np.zeros(n_train, dtype=float)

        coef_path = np.zeros((self.config.max_steps, p), dtype=float)
        step_update_l1_path = np.zeros(self.config.max_steps, dtype=float)
        step_size_path = np.zeros(self.config.max_steps, dtype=float)
        mean_instability_path = np.zeros(self.config.max_steps, dtype=float)
        support_score_path = np.zeros((self.config.max_steps, p), dtype=float)
        consensus_weight_path = np.zeros((self.config.max_steps, p), dtype=float)
        conditional_mean_gamma_path = np.zeros((self.config.max_steps, p), dtype=float)
        group_support_score_path = np.zeros((self.config.max_steps, n_groups), dtype=float)
        sign_consistency_path = np.zeros((self.config.max_steps, p), dtype=float)
        selected_feature_matrix = np.zeros((self.config.max_steps, self.config.n_inner_bootstraps), dtype=int)
        selection_frequency_path = np.zeros((self.config.max_steps, p), dtype=float)
        cumulative_feature_counts = np.zeros(p, dtype=float)
        eps = 1e-12

        for step in range(self.config.max_steps):
            residual = y_train_centered - pred
            sample_weight = _residual_sampling_weights(
                residual,
                self.config.residual_weight_power,
                self.config.residual_weight_eps,
            )

            bootstrap_features = np.zeros(self.config.n_inner_bootstraps, dtype=int)
            bootstrap_gammas = np.zeros(self.config.n_inner_bootstraps, dtype=float)
            bootstrap_predictions = np.zeros((self.config.n_inner_bootstraps, n_train), dtype=float)

            for bag_idx in range(self.config.n_inner_bootstraps):
                sample_idx = rng.choice(n_train, size=n_train, replace=True, p=sample_weight)
                feature_idx, gamma = _fit_componentwise_bootstrap_learner(
                    X_train_std[sample_idx],
                    residual[sample_idx],
                )
                bootstrap_features[bag_idx] = int(feature_idx)
                bootstrap_gammas[bag_idx] = float(gamma)
                bootstrap_predictions[bag_idx] = float(gamma) * X_train_std[:, feature_idx]
                selected_feature_matrix[step, bag_idx] = int(feature_idx)
                cumulative_feature_counts[feature_idx] += 1.0

            selection_frequency_step = np.zeros(p, dtype=float)
            sign_consistency_step = np.zeros(p, dtype=float)
            conditional_mean_gamma_step = np.zeros(p, dtype=float)
            gamma_variance_step = np.zeros(p, dtype=float)

            unique_features, counts = np.unique(bootstrap_features, return_counts=True)
            for feature_idx, count in zip(unique_features.astype(int), counts.astype(int)):
                mask = bootstrap_features == feature_idx
                gamma_j = bootstrap_gammas[mask]
                count_j = float(count)
                selection_frequency_step[feature_idx] = count_j / float(self.config.n_inner_bootstraps)
                conditional_mean_gamma_step[feature_idx] = float(np.mean(gamma_j))
                sign_consistency_step[feature_idx] = float(np.abs(np.mean(np.sign(gamma_j))))
                gamma_variance_step[feature_idx] = float(np.mean((gamma_j - conditional_mean_gamma_step[feature_idx]) ** 2))

            group_selection_frequency = np.zeros(n_groups, dtype=float)
            group_sign_consistency = np.zeros(n_groups, dtype=float)
            group_mean_gamma = np.zeros(n_groups, dtype=float)
            group_gamma_variance = np.zeros(n_groups, dtype=float)
            for group_idx, members in enumerate(feature_groups):
                member_mask = np.isin(bootstrap_features, members)
                if not np.any(member_mask):
                    continue
                gamma_g = bootstrap_gammas[member_mask]
                group_selection_frequency[group_idx] = float(np.mean(member_mask))
                group_mean_gamma[group_idx] = float(np.mean(gamma_g))
                group_sign_consistency[group_idx] = float(np.abs(np.mean(np.sign(gamma_g))))
                group_gamma_variance[group_idx] = float(np.mean((gamma_g - group_mean_gamma[group_idx]) ** 2))
            consensus_weight = np.zeros(p, dtype=float)
            active = selection_frequency_step > 0.0
            if np.any(active):
                frequency_gate = np.ones(np.count_nonzero(active), dtype=float)
                sign_gate = np.ones(np.count_nonzero(active), dtype=float)
                if self.config.min_consensus_frequency > 0.0:
                    frequency_gate = np.minimum(
                        selection_frequency_step[active] / float(self.config.min_consensus_frequency),
                        1.0,
                    )
                if self.config.min_sign_consistency > 0.0:
                    sign_gate = np.minimum(
                        sign_consistency_step[active] / float(self.config.min_sign_consistency),
                        1.0,
                    )
                frequency_factor = np.power(selection_frequency_step[active], self.config.consensus_frequency_power)
                sign_factor = np.power(sign_consistency_step[active], self.config.consensus_sign_power)
                instability_factor = np.power(
                    1.0 + self.config.instability_lambda * gamma_variance_step[active],
                    self.config.instability_power,
                )
                consensus_weight[active] = (
                    frequency_gate * sign_gate * frequency_factor * sign_factor / np.maximum(instability_factor, eps)
                )

            direction_coef = consensus_weight * conditional_mean_gamma_step
            if self.config.enable_group_consensus:
                group_weight = np.zeros(n_groups, dtype=float)
                active_groups = group_selection_frequency > 0.0
                if np.any(active_groups):
                    group_frequency_gate = np.ones(np.count_nonzero(active_groups), dtype=float)
                    group_sign_gate = np.ones(np.count_nonzero(active_groups), dtype=float)
                    if self.config.min_consensus_frequency > 0.0:
                        group_frequency_gate = np.minimum(
                            group_selection_frequency[active_groups] / float(self.config.min_consensus_frequency),
                            1.0,
                        )
                    if self.config.min_sign_consistency > 0.0:
                        group_sign_gate = np.minimum(
                            group_sign_consistency[active_groups] / float(self.config.min_sign_consistency),
                            1.0,
                        )
                    group_frequency_factor = np.power(
                        group_selection_frequency[active_groups],
                        self.config.consensus_frequency_power,
                    )
                    group_sign_factor = np.power(
                        group_sign_consistency[active_groups],
                        self.config.consensus_sign_power,
                    )
                    group_instability_factor = np.power(
                        1.0 + self.config.instability_lambda * group_gamma_variance[active_groups],
                        self.config.instability_power,
                    )
                    group_weight[active_groups] = (
                        group_frequency_gate
                        * group_sign_gate
                        * group_frequency_factor
                        * group_sign_factor
                        / np.maximum(group_instability_factor, eps)
                    )

                direction_coef = np.zeros(p, dtype=float)
                group_residual_alignment = np.abs(X_train_std.T @ residual)
                for group_idx, members in enumerate(feature_groups):
                    if group_weight[group_idx] <= 0.0:
                        continue
                    representative = int(members[np.argmax(group_residual_alignment[members])])
                    direction_coef[representative] = group_weight[group_idx] * group_mean_gamma[group_idx]
                    consensus_weight[representative] = group_weight[group_idx]
                    conditional_mean_gamma_step[representative] = group_mean_gamma[group_idx]
                    sign_consistency_step[representative] = group_sign_consistency[group_idx]
            else:
                group_weight = np.zeros(n_groups, dtype=float)
            direction_pred = X_train_std @ direction_coef
            consensus_prediction = np.mean(bootstrap_predictions, axis=0)
            prediction_instability = np.mean(
                (bootstrap_predictions - consensus_prediction[None, :]) ** 2,
                axis=0,
            )
            numerator = float(np.dot(residual, direction_pred))
            denom_main = (1.0 / self.config.eta) * float(np.dot(direction_pred, direction_pred))
            denom_penalty = self.config.instability_lambda * float(np.sum(prediction_instability))
            denominator = max(denom_main + denom_penalty, eps)
            alpha = numerator / denominator if np.any(active) else 0.0

            delta_coef = alpha * direction_coef
            delta_pred = alpha * direction_pred

            coef = coef + delta_coef
            pred = pred + delta_pred

            support_evidence = consensus_weight * np.abs(conditional_mean_gamma_step)
            group_support_evidence = group_weight * np.abs(group_mean_gamma)
            if self.config.enable_group_consensus:
                support_evidence = np.zeros(p, dtype=float)
                for group_idx, members in enumerate(feature_groups):
                    if group_support_evidence[group_idx] <= 0.0:
                        continue
                    support_evidence[members] = np.maximum(
                        support_evidence[members],
                        group_support_evidence[group_idx],
                    )
            if step == 0:
                support_score_path[step] = support_evidence
                group_support_score_path[step] = group_support_evidence
            else:
                support_score_path[step] = (step * support_score_path[step - 1] + support_evidence) / float(step + 1)
                group_support_score_path[step] = (
                    step * group_support_score_path[step - 1] + group_support_evidence
                ) / float(step + 1)
            coef_path[step] = coef
            step_update_l1_path[step] = float(np.sum(np.abs(delta_coef)))
            step_size_path[step] = float(alpha)
            mean_instability_path[step] = float(np.mean(prediction_instability))
            consensus_weight_path[step] = consensus_weight
            conditional_mean_gamma_path[step] = conditional_mean_gamma_step
            sign_consistency_path[step] = sign_consistency_step
            selection_frequency_path[step] = cumulative_feature_counts / float((step + 1) * self.config.n_inner_bootstraps)

        self.coef_path_ = coef_path
        self.step_update_l1_path_ = step_update_l1_path
        self.step_size_path_ = step_size_path
        self.support_score_path_ = support_score_path
        self.group_support_score_path_ = group_support_score_path
        self.mean_instability_path_ = mean_instability_path
        self.consensus_weight_path_ = consensus_weight_path
        self.conditional_mean_gamma_path_ = conditional_mean_gamma_path
        self.sign_consistency_path_ = sign_consistency_path
        self.selected_feature_matrix_ = selected_feature_matrix
        self.selection_frequency_path_ = selection_frequency_path

        valid_pred_path = _predict_centered_from_coef_path(X_valid_std, coef_path)
        rows: List[Dict[str, float]] = []
        best_step = None
        best_valid_mse = None
        for step in self.selection_checkpoints:
            pred_valid = valid_pred_path[step - 1] + stats.y_mean
            mse = float(mean_squared_error(valid_split.y, pred_valid))
            selection_freq_step = selection_frequency_path[step - 1]
            support_size = int(
                np.count_nonzero(selection_freq_step >= self.config.support_frequency_threshold)
            )
            rows.append(
                {
                    "checkpoint": int(step),
                    "valid_mse": mse,
                    "support_size": float(support_size),
                    "step_update_l1": float(step_update_l1_path[step - 1]),
                    "step_size_alpha": float(step_size_path[step - 1]),
                    "mean_instability": float(mean_instability_path[step - 1]),
                    "mean_consensus_weight": float(np.mean(consensus_weight_path[step - 1])),
                    "mean_support_score": float(np.mean(support_score_path[step - 1])),
                    "group_support_size": float(
                        np.count_nonzero(group_support_score_path[step - 1] >= self.config.support_frequency_threshold)
                    ),
                    "mean_sign_consistency": float(np.mean(sign_consistency_path[step - 1])),
                }
            )
            if best_valid_mse is None or mse < best_valid_mse:
                best_valid_mse = mse
                best_step = int(step)

        self.selection_trace_ = pd.DataFrame(rows)
        self.selected_step_ = int(best_step)
        self.selected_coef_ = coef_path[self.selected_step_ - 1].copy()
        self.support_score_ = support_score_path[self.selected_step_ - 1].copy()
        self.selection_frequency_ = selection_frequency_path[self.selected_step_ - 1].copy()
        self.selected_support_from_frequency_ = np.flatnonzero(
            self.selection_frequency_ >= self.config.support_frequency_threshold
        ).astype(int)
        self.selected_support_ = np.flatnonzero(
            self.support_score_ >= self.config.support_frequency_threshold
        ).astype(int)
        return self

    def coef_at_step(self, step: Optional[int] = None) -> Array:
        if self.coef_path_ is None:
            raise RuntimeError("Model has not been fit")
        use_step = int(step or self.selected_step_ or self.config.max_steps)
        _validate_step(use_step, self.config.max_steps)
        return self.coef_path_[use_step - 1].copy()

    def predict(self, X: Array, step: Optional[int] = None) -> Array:
        stats = _require_standardization(self.standardization_)
        coef = self.coef_at_step(step=step)
        X_std = _transform_features(X, stats)
        return X_std @ coef + stats.y_mean

    def predict_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.coef_path_ is None:
            raise RuntimeError("Model has not been fit")
        stats = _require_standardization(self.standardization_)
        X_std = _transform_features(X, stats)
        out: Dict[int, Array] = {}
        for step in sorted({int(s) for s in checkpoints}):
            _validate_step(step, self.config.max_steps)
            out[int(step)] = X_std @ self.coef_path_[step - 1] + stats.y_mean
        return out

    def support_at_step(self, step: Optional[int] = None) -> Array:
        if self.support_score_path_ is None:
            raise RuntimeError("Model has not been fit")
        use_step = int(step or self.selected_step_ or self.config.max_steps)
        _validate_step(use_step, self.config.max_steps)
        support_score = self.support_score_path_[use_step - 1]
        return np.flatnonzero(
            support_score >= self.config.support_frequency_threshold
        ).astype(int)

    def support_score_at_step(self, step: Optional[int] = None) -> Array:
        if self.support_score_path_ is None:
            raise RuntimeError("Model has not been fit")
        use_step = int(step or self.selected_step_ or self.config.max_steps)
        _validate_step(use_step, self.config.max_steps)
        return self.support_score_path_[use_step - 1].copy()

    def group_support_at_step(self, step: Optional[int] = None) -> List[Array]:
        if self.group_support_score_path_ is None or self.feature_groups_ is None:
            raise RuntimeError("Model has not been fit")
        use_step = int(step or self.selected_step_ or self.config.max_steps)
        _validate_step(use_step, self.config.max_steps)
        group_scores = self.group_support_score_path_[use_step - 1]
        active_groups = np.flatnonzero(
            group_scores >= self.config.support_frequency_threshold
        ).astype(int)
        return [self.feature_groups_[idx].copy() for idx in active_groups]

    def group_support_score_at_step(self, step: Optional[int] = None) -> Array:
        if self.group_support_score_path_ is None:
            raise RuntimeError("Model has not been fit")
        use_step = int(step or self.selected_step_ or self.config.max_steps)
        _validate_step(use_step, self.config.max_steps)
        return self.group_support_score_path_[use_step - 1].copy()

    def feature_groups(self) -> List[Array]:
        if self.feature_groups_ is None:
            raise RuntimeError("Model has not been fit")
        return [group.copy() for group in self.feature_groups_]

    def support_from_frequency_at_step(self, step: Optional[int] = None) -> Array:
        if self.selection_frequency_path_ is None:
            raise RuntimeError("Model has not been fit")
        use_step = int(step or self.selected_step_ or self.config.max_steps)
        _validate_step(use_step, self.config.max_steps)
        selection_freq = self.selection_frequency_path_[use_step - 1]
        return np.flatnonzero(
            selection_freq >= self.config.support_frequency_threshold
        ).astype(int)

    def selection_frequency_at_step(self, step: Optional[int] = None) -> Array:
        if self.selection_frequency_path_ is None:
            raise RuntimeError("Model has not been fit")
        use_step = int(step or self.selected_step_ or self.config.max_steps)
        _validate_step(use_step, self.config.max_steps)
        return self.selection_frequency_path_[use_step - 1].copy()

class LassoPathRegressorWrapper(SparseRegressionWrapperBase):
    def __init__(self, config: LassoPathConfig) -> None:
        super().__init__()
        self.config = config
        self.alpha_path_: Optional[Array] = None
        self.coef_path_: Optional[Array] = None
        self.selected_alpha_: Optional[float] = None
        self.selected_coef_: Optional[Array] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(
        self,
        train_split: SparseRegressionSplit,
        valid_split: SparseRegressionSplit,
    ) -> "LassoPathRegressorWrapper":
        X_train_std, y_train_centered, stats = _prepare_train_arrays(train_split.X, train_split.y)
        X_valid_std = _transform_features(valid_split.X, stats)
        self.standardization_ = stats

        alphas = None if self.config.alphas is None else np.asarray(self.config.alphas, dtype=float)
        alpha_path, coef_path, _ = lasso_path(
            X=X_train_std,
            y=y_train_centered,
            alphas=alphas,
            n_alphas=self.config.n_alphas,
            eps=self.config.eps,
        )
        self.alpha_path_ = np.asarray(alpha_path, dtype=float)
        self.coef_path_ = np.asarray(coef_path.T, dtype=float)

        rows: List[Dict[str, float]] = []
        best_idx = None
        best_valid_mse = None
        for idx, alpha in enumerate(self.alpha_path_):
            pred = X_valid_std @ self.coef_path_[idx] + stats.y_mean
            mse = float(mean_squared_error(valid_split.y, pred))
            support_size = int(np.count_nonzero(np.abs(self.coef_path_[idx]) > self.config.coef_tol))
            rows.append(
                {
                    "alpha": float(alpha),
                    "valid_mse": mse,
                    "support_size": float(support_size),
                }
            )
            if best_valid_mse is None or mse < best_valid_mse:
                best_valid_mse = mse
                best_idx = idx

        self.selection_trace_ = pd.DataFrame(rows)
        self.selected_alpha_ = float(self.alpha_path_[best_idx])
        self.selected_coef_ = self.coef_path_[best_idx].copy()
        self.selected_support_ = np.flatnonzero(np.abs(self.selected_coef_) > self.config.coef_tol).astype(int)
        return self

    def predict(self, X: Array, alpha: Optional[float] = None) -> Array:
        if self.coef_path_ is None or self.alpha_path_ is None:
            raise RuntimeError("Model has not been fit")
        stats = _require_standardization(self.standardization_)
        X_std = _transform_features(X, stats)
        if alpha is None:
            coef = self.selected_coef_
        else:
            idx = int(np.argmin(np.abs(self.alpha_path_ - float(alpha))))
            coef = self.coef_path_[idx]
        return X_std @ coef + stats.y_mean

    def support_at_alpha(self, alpha: Optional[float] = None) -> Array:
        if self.coef_path_ is None or self.alpha_path_ is None:
            raise RuntimeError("Model has not been fit")
        if alpha is None:
            coef = self.selected_coef_
        else:
            idx = int(np.argmin(np.abs(self.alpha_path_ - float(alpha))))
            coef = self.coef_path_[idx]
        return np.flatnonzero(np.abs(coef) > self.config.coef_tol).astype(int)


class XGBTreeRegressorWrapper(SparseRegressionWrapperBase):
    def __init__(
        self,
        config: XGBTreeConfig,
        selection_checkpoints: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.selection_checkpoints = _resolve_checkpoints(
            selection_checkpoints,
            max_checkpoint=config.n_estimators,
            default_points=min(25, config.n_estimators),
        )
        self.model = None
        self.selected_checkpoint_: Optional[int] = None
        self.feature_importances_: Optional[Array] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(
        self,
        train_split: SparseRegressionSplit,
        valid_split: SparseRegressionSplit,
    ) -> "XGBTreeRegressorWrapper":
        if XGBRegressor is None:  # pragma: no cover
            raise ImportError("xgboost is not installed, but XGBTreeRegressorWrapper was requested")
        self.model = XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0,
        )
        self.model.fit(train_split.X, train_split.y)

        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_valid_mse = None
        for checkpoint in self.selection_checkpoints:
            pred = np.asarray(self.model.predict(valid_split.X, iteration_range=(0, int(checkpoint))), dtype=float)
            mse = float(mean_squared_error(valid_split.y, pred))
            rows.append(
                {
                    "checkpoint": int(checkpoint),
                    "valid_mse": mse,
                }
            )
            if best_valid_mse is None or mse < best_valid_mse:
                best_valid_mse = mse
                best_checkpoint = int(checkpoint)

        self.selection_trace_ = pd.DataFrame(rows)
        self.selected_checkpoint_ = int(best_checkpoint)
        self.feature_importances_ = np.asarray(self.model.feature_importances_, dtype=float)
        self.selected_support_ = np.flatnonzero(self.feature_importances_ > 0).astype(int)
        return self

    def predict(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        return np.asarray(self.model.predict(X, iteration_range=(0, use_checkpoint)), dtype=float)

    def predict_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        out: Dict[int, Array] = {}
        for checkpoint in sorted({int(c) for c in checkpoints}):
            _validate_step(checkpoint, self.config.n_estimators)
            out[int(checkpoint)] = np.asarray(self.model.predict(X, iteration_range=(0, checkpoint)), dtype=float)
        return out

    def topk_support(self, k: int) -> Array:
        if self.feature_importances_ is None:
            raise RuntimeError("Model has not been fit")
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(int(k), self.feature_importances_.shape[0])
        order = np.argsort(-self.feature_importances_)
        return np.sort(order[:k].astype(int))


class CTBTreeRegressorWrapper(SparseRegressionWrapperBase):
    def __init__(
        self,
        config: CTBTreeConfig,
        selection_checkpoints: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.selection_checkpoints = _resolve_checkpoints(
            selection_checkpoints,
            max_checkpoint=config.n_estimators,
            default_points=min(25, config.n_estimators),
        )
        self.model = None
        self.selected_checkpoint_: Optional[int] = None
        self.feature_importances_: Optional[Array] = None
        self.feature_importances_by_checkpoint_: Optional[Dict[int, Array]] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(
        self,
        train_split: SparseRegressionSplit,
        valid_split: SparseRegressionSplit,
    ) -> "CTBTreeRegressorWrapper":
        self.model = ConsensusTransportBoosting(
            task_type="regression",
            n_estimators=self.config.n_estimators,
            n_inner_bootstraps=self.config.n_inner_bootstraps,
            eta=self.config.eta,
            instability_penalty=self.config.instability_penalty,
            weight_power=self.config.weight_power,
            weight_eps=self.config.weight_eps,
            update_target_mode=self.config.update_target_mode,
            transport_curvature_eps=self.config.transport_curvature_eps,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self.model.fit(train_split.X, train_split.y)

        staged_valid = self.model.decision_function_staged(valid_split.X, checkpoints=self.selection_checkpoints)
        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_valid_mse = None
        for checkpoint in self.selection_checkpoints:
            pred = np.asarray(staged_valid[int(checkpoint)], dtype=float)
            mse = float(mean_squared_error(valid_split.y, pred))
            rows.append({"checkpoint": int(checkpoint), "valid_mse": mse})
            if best_valid_mse is None or mse < best_valid_mse:
                best_valid_mse = mse
                best_checkpoint = int(checkpoint)
        self.selection_trace_ = pd.DataFrame(rows)
        self.selected_checkpoint_ = int(best_checkpoint)

        p = train_split.X.shape[1]
        requested = set(self.selection_checkpoints)
        importances_by_checkpoint: Dict[int, Array] = {}
        cumulative = np.zeros(p, dtype=float)
        learner_count = 0
        for round_idx, round_learners in enumerate(self.model.learners_, start=1):
            for learner in round_learners:
                cumulative += np.asarray(learner.feature_importances_, dtype=float)
                learner_count += 1
            if round_idx in requested:
                if learner_count > 0:
                    importances_by_checkpoint[int(round_idx)] = cumulative / float(learner_count)
                else:
                    importances_by_checkpoint[int(round_idx)] = np.zeros(p, dtype=float)
        self.feature_importances_by_checkpoint_ = importances_by_checkpoint
        self.feature_importances_ = np.asarray(importances_by_checkpoint[self.selected_checkpoint_], dtype=float)
        self.selected_support_ = np.flatnonzero(self.feature_importances_ > 0).astype(int)
        return self

    def predict(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        staged = self.model.decision_function_staged(X, checkpoints=[use_checkpoint])
        return np.asarray(staged[use_checkpoint], dtype=float)

    def predict_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        return {
            int(k): np.asarray(v, dtype=float)
            for k, v in self.model.decision_function_staged(X, checkpoints=checkpoints).items()
        }

    def topk_support(self, k: int) -> Array:
        if self.feature_importances_ is None:
            raise RuntimeError("Model has not been fit")
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(int(k), self.feature_importances_.shape[0])
        order = np.argsort(-self.feature_importances_)
        return np.sort(order[:k].astype(int))


def _reselect_binary_l2boost(model: "L2BoostingClassifierWrapper", valid_split: SparseRegressionSplit) -> None:
    if model.coef_path_ is None or model.selected_feature_path_ is None:
        raise RuntimeError("Model has not been fit")
    staged_scores = model.predict_score_staged(valid_split.X, checkpoints=model.selection_checkpoints)
    rows: List[Dict[str, float]] = []
    best_step = None
    best_log_loss = None
    for step in model.selection_checkpoints:
        prob = _sigmoid_probability(staged_scores[int(step)])
        logloss = _binary_log_loss(valid_split.y, prob)
        acc = _binary_accuracy(valid_split.y, prob)
        coef = model.coef_path_[step - 1]
        support_size = int(np.count_nonzero(np.abs(coef) > model.config.coef_tol))
        rows.append({
            "checkpoint": int(step),
            "valid_log_loss": logloss,
            "valid_accuracy": acc,
            "support_size": float(support_size),
            "last_selected_feature": float(model.selected_feature_path_[step - 1]),
        })
        if best_log_loss is None or logloss < best_log_loss:
            best_log_loss = logloss
            best_step = int(step)
    model.selection_trace_ = pd.DataFrame(rows)
    model.selected_step_ = int(best_step)
    model.selected_coef_ = model.coef_path_[model.selected_step_ - 1].copy()
    model.selected_support_ = np.flatnonzero(np.abs(model.selected_coef_) > model.config.coef_tol).astype(int)


def _reselect_binary_bagged(model: "BaggedComponentwiseClassifierWrapper", valid_split: SparseRegressionSplit) -> None:
    if model.bag_coef_matrix_ is None:
        raise RuntimeError("Model has not been fit")
    staged_scores = model.predict_score_staged(valid_split.X, checkpoints=model.selection_checkpoints)
    rows: List[Dict[str, float]] = []
    best_checkpoint = None
    best_log_loss = None
    mean_support_size_by_checkpoint: Dict[int, float] = {}
    for checkpoint in model.selection_checkpoints:
        prob = _sigmoid_probability(staged_scores[int(checkpoint)])
        logloss = _binary_log_loss(valid_split.y, prob)
        acc = _binary_accuracy(valid_split.y, prob)
        selection_freq = np.mean(np.abs(model.bag_coef_matrix_[:checkpoint]) > model.config.coef_tol, axis=0)
        threshold_support = int(np.count_nonzero(selection_freq >= model.config.support_frequency_threshold))
        mean_support = float(np.mean(np.count_nonzero(np.abs(model.bag_coef_matrix_[:checkpoint]) > model.config.coef_tol, axis=1)))
        mean_support_size_by_checkpoint[int(checkpoint)] = mean_support
        rows.append({
            "checkpoint": int(checkpoint),
            "valid_log_loss": logloss,
            "valid_accuracy": acc,
            "support_size_frequency_threshold": float(threshold_support),
            "mean_single_model_support_size": mean_support,
            "union_support_size": float(np.count_nonzero(np.any(np.abs(model.bag_coef_matrix_[:checkpoint]) > model.config.coef_tol, axis=0))),
        })
        if best_log_loss is None or logloss < best_log_loss:
            best_log_loss = logloss
            best_checkpoint = int(checkpoint)
    model.selection_trace_ = pd.DataFrame(rows)
    model.selected_checkpoint_ = int(best_checkpoint)
    model.selected_coef_ = model.bag_coef_matrix_[: model.selected_checkpoint_].mean(axis=0)
    model.selection_frequency_ = np.mean(np.abs(model.bag_coef_matrix_[: model.selected_checkpoint_]) > model.config.coef_tol, axis=0)
    model.average_abs_coef_ = np.mean(np.abs(model.bag_coef_matrix_[: model.selected_checkpoint_]), axis=0)
    model.selected_support_ = np.flatnonzero(model.selection_frequency_ >= model.config.support_frequency_threshold).astype(int)
    model.mean_support_size_by_checkpoint_ = mean_support_size_by_checkpoint


def _reselect_binary_ctb_sparse(model: "CTBSparseClassifierWrapper", valid_split: SparseRegressionSplit) -> None:
    if model.coef_path_ is None or model.support_score_path_ is None or model.selection_frequency_path_ is None:
        raise RuntimeError("Model has not been fit")
    staged_scores = model.predict_score_staged(valid_split.X, checkpoints=model.selection_checkpoints)
    rows: List[Dict[str, float]] = []
    best_step = None
    best_log_loss = None
    for step in model.selection_checkpoints:
        prob = _sigmoid_probability(staged_scores[int(step)])
        logloss = _binary_log_loss(valid_split.y, prob)
        acc = _binary_accuracy(valid_split.y, prob)
        selection_freq_step = model.selection_frequency_path_[step - 1]
        support_size = int(np.count_nonzero(selection_freq_step >= model.config.support_frequency_threshold))
        rows.append({
            "checkpoint": int(step),
            "valid_log_loss": logloss,
            "valid_accuracy": acc,
            "support_size": float(support_size),
            "mean_selection_frequency": float(np.mean(selection_freq_step)),
            "mean_support_score": float(np.mean(model.support_score_path_[step - 1])),
        })
        if best_log_loss is None or logloss < best_log_loss:
            best_log_loss = logloss
            best_step = int(step)
    model.selection_trace_ = pd.DataFrame(rows)
    model.selected_step_ = int(best_step)
    model.selected_coef_ = model.coef_path_[model.selected_step_ - 1].copy()
    model.support_score_ = model.support_score_path_[model.selected_step_ - 1].copy()
    model.selection_frequency_ = model.selection_frequency_path_[model.selected_step_ - 1].copy()
    model.selected_support_from_frequency_ = np.flatnonzero(model.selection_frequency_ >= model.config.support_frequency_threshold).astype(int)
    model.selected_support_ = np.flatnonzero(model.support_score_ >= model.config.support_frequency_threshold).astype(int)


def _reselect_binary_lasso(model: "LassoPathClassifierWrapper", valid_split: SparseRegressionSplit) -> None:
    if model.coef_path_ is None or model.alpha_path_ is None:
        raise RuntimeError("Model has not been fit")
    rows: List[Dict[str, float]] = []
    best_idx = None
    best_log_loss = None
    for idx, alpha in enumerate(model.alpha_path_):
        score = model.predict_score(valid_split.X, alpha=float(alpha))
        prob = _sigmoid_probability(score)
        logloss = _binary_log_loss(valid_split.y, prob)
        acc = _binary_accuracy(valid_split.y, prob)
        support_size = int(np.count_nonzero(np.abs(model.coef_path_[idx]) > model.config.coef_tol))
        rows.append({
            "alpha": float(alpha),
            "valid_log_loss": logloss,
            "valid_accuracy": acc,
            "support_size": float(support_size),
        })
        if best_log_loss is None or logloss < best_log_loss:
            best_log_loss = logloss
            best_idx = idx
    model.selection_trace_ = pd.DataFrame(rows)
    model.selected_alpha_ = float(model.alpha_path_[best_idx])
    model.selected_coef_ = model.coef_path_[best_idx].copy()
    model.selected_support_ = np.flatnonzero(np.abs(model.selected_coef_) > model.config.coef_tol).astype(int)


class L2BoostingClassifierWrapper(L2BoostingRegressorWrapper, SparseClassificationWrapperBase):
    def fit(self, train_split: SparseRegressionSplit, valid_split: SparseRegressionSplit) -> "L2BoostingClassifierWrapper":
        super().fit(train_split, valid_split)
        _reselect_binary_l2boost(self, valid_split)
        return self

    def predict_score(self, X: Array, step: Optional[int] = None) -> Array:
        return L2BoostingRegressorWrapper.predict(self, X, step=step)

    def predict_score_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        return L2BoostingRegressorWrapper.predict_staged(self, X, checkpoints=checkpoints)

    def predict_proba(self, X: Array, step: Optional[int] = None) -> Array:
        return _sigmoid_probability(self.predict_score(X, step=step))


class BaggedComponentwiseClassifierWrapper(BaggedComponentwiseRegressorWrapper, SparseClassificationWrapperBase):
    def fit(self, train_split: SparseRegressionSplit, valid_split: SparseRegressionSplit) -> "BaggedComponentwiseClassifierWrapper":
        super().fit(train_split, valid_split)
        _reselect_binary_bagged(self, valid_split)
        return self

    def predict_score(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        return BaggedComponentwiseRegressorWrapper.predict(self, X, checkpoint=checkpoint)

    def predict_score_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        return BaggedComponentwiseRegressorWrapper.predict_staged(self, X, checkpoints=checkpoints)

    def predict_proba(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        return _sigmoid_probability(self.predict_score(X, checkpoint=checkpoint))


class CTBSparseClassifierWrapper(CTBSparseRegressorWrapper, SparseClassificationWrapperBase):
    def fit(self, train_split: SparseRegressionSplit, valid_split: SparseRegressionSplit) -> "CTBSparseClassifierWrapper":
        super().fit(train_split, valid_split)
        _reselect_binary_ctb_sparse(self, valid_split)
        return self

    def predict_score(self, X: Array, step: Optional[int] = None) -> Array:
        return CTBSparseRegressorWrapper.predict(self, X, step=step)

    def predict_score_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        return CTBSparseRegressorWrapper.predict_staged(self, X, checkpoints=checkpoints)

    def predict_proba(self, X: Array, step: Optional[int] = None) -> Array:
        return _sigmoid_probability(self.predict_score(X, step=step))


class LassoPathClassifierWrapper(LassoPathRegressorWrapper, SparseClassificationWrapperBase):
    def fit(self, train_split: SparseRegressionSplit, valid_split: SparseRegressionSplit) -> "LassoPathClassifierWrapper":
        super().fit(train_split, valid_split)
        _reselect_binary_lasso(self, valid_split)
        return self

    def predict_score(self, X: Array, alpha: Optional[float] = None) -> Array:
        return LassoPathRegressorWrapper.predict(self, X, alpha=alpha)

    def predict_proba(self, X: Array, alpha: Optional[float] = None) -> Array:
        return _sigmoid_probability(self.predict_score(X, alpha=alpha))


class XGBTreeClassifierWrapper(SparseClassificationWrapperBase):
    def __init__(self, config: XGBTreeConfig, selection_checkpoints: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.config = config
        self.selection_checkpoints = _resolve_checkpoints(selection_checkpoints, max_checkpoint=config.n_estimators, default_points=min(25, config.n_estimators))
        self.model = None
        self.selected_checkpoint_: Optional[int] = None
        self.feature_importances_: Optional[Array] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(self, train_split: SparseRegressionSplit, valid_split: SparseRegressionSplit) -> "XGBTreeClassifierWrapper":
        if XGBClassifier is None:  # pragma: no cover
            raise ImportError("xgboost is not installed, but XGBTreeClassifierWrapper was requested")
        self.model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=self.config.reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0,
        )
        self.model.fit(train_split.X, np.asarray(train_split.y, dtype=int))
        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_valid_log_loss = None
        for checkpoint in self.selection_checkpoints:
            prob = np.asarray(self.model.predict_proba(valid_split.X, iteration_range=(0, int(checkpoint)))[:, 1], dtype=float)
            logloss = _binary_log_loss(valid_split.y, prob)
            acc = _binary_accuracy(valid_split.y, prob)
            rows.append({"checkpoint": int(checkpoint), "valid_log_loss": logloss, "valid_accuracy": acc})
            if best_valid_log_loss is None or logloss < best_valid_log_loss:
                best_valid_log_loss = logloss
                best_checkpoint = int(checkpoint)
        self.selection_trace_ = pd.DataFrame(rows)
        self.selected_checkpoint_ = int(best_checkpoint)
        self.feature_importances_ = np.asarray(self.model.feature_importances_, dtype=float)
        self.selected_support_ = np.flatnonzero(self.feature_importances_ > 0).astype(int)
        return self

    def predict_proba(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        return np.asarray(self.model.predict_proba(X, iteration_range=(0, use_checkpoint))[:, 1], dtype=float)

    def predict_proba_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        out: Dict[int, Array] = {}
        for checkpoint in sorted({int(c) for c in checkpoints}):
            _validate_step(checkpoint, self.config.n_estimators)
            out[int(checkpoint)] = np.asarray(self.model.predict_proba(X, iteration_range=(0, checkpoint))[:, 1], dtype=float)
        return out

    def topk_support(self, k: int) -> Array:
        if self.feature_importances_ is None:
            raise RuntimeError("Model has not been fit")
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(int(k), self.feature_importances_.shape[0])
        order = np.argsort(-self.feature_importances_)
        return np.sort(order[:k].astype(int))


class CTBTreeClassifierWrapper(SparseClassificationWrapperBase):
    def __init__(self, config: CTBTreeConfig, selection_checkpoints: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.config = config
        self.selection_checkpoints = _resolve_checkpoints(selection_checkpoints, max_checkpoint=config.n_estimators, default_points=min(25, config.n_estimators))
        self.model = None
        self.selected_checkpoint_: Optional[int] = None
        self.feature_importances_: Optional[Array] = None
        self.feature_importances_by_checkpoint_: Optional[Dict[int, Array]] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(self, train_split: SparseRegressionSplit, valid_split: SparseRegressionSplit) -> "CTBTreeClassifierWrapper":
        self.model = ConsensusTransportBoosting(
            task_type="classification",
            n_estimators=self.config.n_estimators,
            n_inner_bootstraps=self.config.n_inner_bootstraps,
            eta=self.config.eta,
            instability_penalty=self.config.instability_penalty,
            weight_power=self.config.weight_power,
            weight_eps=self.config.weight_eps,
            update_target_mode=self.config.update_target_mode,
            transport_curvature_eps=self.config.transport_curvature_eps,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self.model.fit(train_split.X, np.asarray(train_split.y, dtype=int))
        staged_valid = self.model.predict_proba_staged(valid_split.X, checkpoints=self.selection_checkpoints)
        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_valid_log_loss = None
        for checkpoint in self.selection_checkpoints:
            prob = np.asarray(staged_valid[int(checkpoint)][:, 1], dtype=float)
            logloss = _binary_log_loss(valid_split.y, prob)
            acc = _binary_accuracy(valid_split.y, prob)
            rows.append({"checkpoint": int(checkpoint), "valid_log_loss": logloss, "valid_accuracy": acc})
            if best_valid_log_loss is None or logloss < best_valid_log_loss:
                best_valid_log_loss = logloss
                best_checkpoint = int(checkpoint)
        self.selection_trace_ = pd.DataFrame(rows)
        self.selected_checkpoint_ = int(best_checkpoint)

        p = train_split.X.shape[1]
        requested = set(self.selection_checkpoints)
        importances_by_checkpoint: Dict[int, Array] = {}
        cumulative = np.zeros(p, dtype=float)
        learner_count = 0
        for round_idx, round_learners in enumerate(self.model.learners_, start=1):
            for learner in round_learners:
                cumulative += np.asarray(learner.feature_importances_, dtype=float)
                learner_count += 1
            if round_idx in requested:
                importances_by_checkpoint[int(round_idx)] = cumulative / float(learner_count) if learner_count > 0 else np.zeros(p, dtype=float)
        self.feature_importances_by_checkpoint_ = importances_by_checkpoint
        self.feature_importances_ = np.asarray(importances_by_checkpoint[self.selected_checkpoint_], dtype=float)
        self.selected_support_ = np.flatnonzero(self.feature_importances_ > 0).astype(int)
        return self

    def predict_proba(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        staged = self.model.predict_proba_staged(X, checkpoints=[use_checkpoint])
        return np.asarray(staged[use_checkpoint][:, 1], dtype=float)

    def predict_proba_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        staged = self.model.predict_proba_staged(X, checkpoints=checkpoints)
        return {int(k): np.asarray(v[:, 1], dtype=float) for k, v in staged.items()}

    def topk_support(self, k: int) -> Array:
        if self.feature_importances_ is None:
            raise RuntimeError("Model has not been fit")
        if k <= 0:
            raise ValueError("k must be positive")
        k = min(int(k), self.feature_importances_.shape[0])
        order = np.argsort(-self.feature_importances_)
        return np.sort(order[:k].astype(int))


def build_experiment4_model(
    model_name: str,
    *,
    task_type: str = "regression",
    random_state: int = 0,
    selection_checkpoints: Optional[Sequence[int]] = None,
    trajectory_checkpoints: Optional[Sequence[int]] = None,
    **kwargs,
):
    name = str(model_name)
    task = str(task_type)
    if task not in {"regression", "classification"}:
        raise ValueError(f"Unsupported task_type={task_type!r}")
    if name == "l2boost":
        config = L2BoostingConfig(random_state=random_state, **kwargs)
        wrapper_cls = L2BoostingRegressorWrapper if task == "regression" else L2BoostingClassifierWrapper
        return wrapper_cls(
            config=config,
            selection_checkpoints=selection_checkpoints,
            trajectory_checkpoints=trajectory_checkpoints,
        )
    if name == "bagged_componentwise":
        config = BaggedComponentwiseConfig(random_state=random_state, **kwargs)
        wrapper_cls = BaggedComponentwiseRegressorWrapper if task == "regression" else BaggedComponentwiseClassifierWrapper
        return wrapper_cls(
            config=config,
            selection_checkpoints=selection_checkpoints,
        )
    if name == "ctb_sparse":
        config = CTBSparseConfig(random_state=random_state, **kwargs)
        wrapper_cls = CTBSparseRegressorWrapper if task == "regression" else CTBSparseClassifierWrapper
        return wrapper_cls(
            config=config,
            selection_checkpoints=selection_checkpoints,
            trajectory_checkpoints=trajectory_checkpoints,
        )
    if name == "ctb_tree":
        config = CTBTreeConfig(random_state=random_state, **kwargs)
        wrapper_cls = CTBTreeRegressorWrapper if task == "regression" else CTBTreeClassifierWrapper
        return wrapper_cls(
            config=config,
            selection_checkpoints=selection_checkpoints,
        )
    if name == "lasso":
        config = LassoPathConfig(random_state=random_state, **kwargs)
        return LassoPathRegressorWrapper(config=config) if task == "regression" else LassoPathClassifierWrapper(config=config)
    if name == "xgb_tree":
        config = XGBTreeConfig(random_state=random_state, **kwargs)
        return XGBTreeRegressorWrapper(config=config, selection_checkpoints=selection_checkpoints) if task == "regression" else XGBTreeClassifierWrapper(config=config, selection_checkpoints=selection_checkpoints)
    raise ValueError(f"Unsupported model_name={model_name!r}")


def default_experiment4_model_grid(random_state: int = 0) -> Dict[str, Dict[str, object]]:
    return {
        "l2boost": {
            "max_steps": 300,
            "learning_rate": 0.1,
            "random_state": random_state,
        },
        "bagged_componentwise": {
            "n_estimators": 100,
            "base_max_steps": 100,
            "learning_rate": 0.1,
            "support_frequency_threshold": 0.5,
            "random_state": random_state,
        },
        "ctb_sparse": {
            "max_steps": 300,
            "n_inner_bootstraps": 8,
            "eta": 1.0,
            "residual_weight_power": 1.0,
            "residual_weight_eps": 1e-8,
            "consensus_frequency_power": 2.0,
            "consensus_sign_power": 1.0,
            "instability_lambda": 1.0,
            "instability_power": 1.0,
            "min_consensus_frequency": 0.25,
            "min_sign_consistency": 0.75,
            "support_frequency_threshold": 0.05,
            "random_state": random_state,
        },
        "ctb_tree": {
            "n_estimators": 300,
            "n_inner_bootstraps": 8,
            "eta": 1.0,
            "max_depth": 1,
            "min_samples_leaf": 5,
            "update_target_mode": "legacy",
            "transport_curvature_eps": 1e-6,
            "random_state": random_state,
        },
        "lasso": {
            "n_alphas": 80,
            "eps": 1e-3,
            "random_state": random_state,
        },
        "xgb_tree": {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": random_state,
        },
    }


def _prepare_train_arrays(X: Array, y: Array) -> Tuple[Array, Array, StandardizationStats]:
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    x_std = np.where(x_std < 1e-12, 1.0, x_std)
    y_mean = float(np.mean(y))
    X_std = (X - x_mean) / x_std
    y_centered = y - y_mean
    stats = StandardizationStats(x_mean=x_mean.astype(float), x_std=x_std.astype(float), y_mean=y_mean)
    return X_std.astype(float), y_centered.astype(float), stats


def _transform_features(X: Array, stats: StandardizationStats) -> Array:
    return ((X - stats.x_mean) / stats.x_std).astype(float)


def _fit_l2boost_path(
    X: Array,
    y_centered: Array,
    max_steps: int,
    learning_rate: float,
) -> L2BoostPathResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not (0.0 < learning_rate <= 1.0):
        raise ValueError("learning_rate must lie in (0, 1]")

    n_samples, p = X.shape
    coef = np.zeros(p, dtype=float)
    pred = np.zeros(n_samples, dtype=float)
    denom = np.sum(X * X, axis=0)
    denom = np.where(denom < 1e-12, 1.0, denom)

    coef_path = np.zeros((max_steps, p), dtype=float)
    train_prediction_path = np.zeros((max_steps, n_samples), dtype=float)
    selected_feature_path = np.zeros(max_steps, dtype=int)

    for step in range(max_steps):
        residual = y_centered - pred
        numer = X.T @ residual
        gamma = numer / denom
        improvement = (numer**2) / denom
        best_feature = int(np.argmax(improvement))
        update = float(learning_rate * gamma[best_feature])
        coef[best_feature] += update
        pred = pred + update * X[:, best_feature]

        coef_path[step] = coef
        train_prediction_path[step] = pred
        selected_feature_path[step] = best_feature

    return L2BoostPathResult(
        coef_path=coef_path,
        selected_feature_path=selected_feature_path,
        train_prediction_path=train_prediction_path,
    )


def _predict_centered_from_coef_path(X_std: Array, coef_path: Array) -> Array:
    return coef_path @ X_std.T


def _compute_entry_step(selected_feature_path: Array, p: int) -> Array:
    entry = np.full(p, fill_value=np.inf, dtype=float)
    for step, feature_idx in enumerate(selected_feature_path, start=1):
        if np.isinf(entry[feature_idx]):
            entry[feature_idx] = float(step)
    return entry

def _fit_componentwise_bootstrap_learner(X: Array, y_centered: Array) -> Tuple[int, float]:
    denom = np.sum(X * X, axis=0)
    denom = np.where(denom < 1e-12, 1.0, denom)
    numer = X.T @ y_centered
    improvement = (numer ** 2) / denom
    best_feature = int(np.argmax(improvement))
    gamma = float(numer[best_feature] / denom[best_feature])
    return best_feature, gamma

def _build_correlation_groups(X_std: Array, corr_threshold: float) -> Tuple[List[Array], Array]:
    p = X_std.shape[1]
    if p == 0:
        return [], np.zeros(0, dtype=int)
    if p == 1:
        return [np.array([0], dtype=int)], np.array([0], dtype=int)

    corr = np.abs(np.corrcoef(X_std, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    visited = np.zeros(p, dtype=bool)
    feature_to_group = np.full(p, -1, dtype=int)
    groups: List[Array] = []

    for start in range(p):
        if visited[start]:
            continue
        stack = [int(start)]
        members: List[int] = []
        visited[start] = True
        while stack:
            node = stack.pop()
            members.append(node)
            neighbors = np.flatnonzero(corr[node] >= corr_threshold)
            for nb in neighbors.astype(int):
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(int(nb))
        group_idx = len(groups)
        groups.append(np.asarray(sorted(members), dtype=int))
        feature_to_group[groups[-1]] = group_idx
    return groups, feature_to_group

def _residual_sampling_weights(residual: Array, weight_power: float, weight_eps: float) -> Array:
    if weight_eps <= 0.0:
        raise ValueError("weight_eps must be positive")
    if weight_power < 0.0:
        raise ValueError("weight_power must be non-negative")
    raw = (np.abs(residual) + float(weight_eps)) ** float(weight_power)
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(residual.shape[0], 1.0 / float(residual.shape[0]), dtype=float)
    return raw / total

def _resolve_checkpoints(
    checkpoints: Optional[Sequence[int]],
    *,
    max_checkpoint: int,
    default_points: int,
) -> List[int]:
    if max_checkpoint <= 0:
        raise ValueError("max_checkpoint must be positive")
    if checkpoints is None:
        default_points = max(1, min(int(default_points), max_checkpoint))
        raw = np.linspace(1, max_checkpoint, num=default_points, dtype=int)
        resolved = sorted({int(x) for x in raw.tolist()} | {1, max_checkpoint})
    else:
        resolved = sorted({int(x) for x in checkpoints if 1 <= int(x) <= max_checkpoint})
    if not resolved:
        raise ValueError("checkpoints resolved to an empty set")
    return resolved


def _validate_step(step: int, max_checkpoint: int) -> None:
    if step <= 0 or step > max_checkpoint:
        raise ValueError(f"checkpoint must lie in [1, {max_checkpoint}], got {step}")


def _require_standardization(stats: Optional[StandardizationStats]) -> StandardizationStats:
    if stats is None:
        raise RuntimeError("Model has not been fit")
    return stats
