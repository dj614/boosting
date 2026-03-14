from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor

try:  # pragma: no cover
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


TaskType = Literal["regression", "classification"]
UpdateTargetMode = Literal["legacy", "loss_aware"]
WeakLearnerBackend = Literal["sklearn_tree", "xgb_tree"]


class ConsensusTransportBoosting(BaseEstimator):
    """Minimal CTB implementation for regression and binary classification.

    The weak learner is a regressor fit to the current pseudo-response. By default
    this is an xgboost single-tree regressor by default; a sklearn
    regression tree can be requested via ``weak_learner_backend='sklearn_tree'``. For binary classification,
    the ensemble state is maintained on the raw score scale and probabilities are
    obtained via a sigmoid link.
    """

    def __init__(
        self,
        *,
        task_type: TaskType = "regression",
        n_estimators: int = 50,
        n_inner_bootstraps: int = 8,
        eta: float = 1.0,
        instability_penalty: float = 0.0,
        weight_power: float = 1.0,
        weight_eps: float = 1e-8,
        update_target_mode: UpdateTargetMode = "legacy",
        transport_curvature_eps: float = 1e-6,
        denom_eps: float = 1e-12,
        max_depth: int | None = 1,
        min_samples_leaf: int = 5,
        weak_learner_backend: WeakLearnerBackend = "xgb_tree",
        xgb_learning_rate: float = 0.1,
        xgb_subsample: float = 1.0,
        xgb_colsample_bytree: float = 0.8,
        xgb_reg_lambda: float = 1.0,
        xgb_min_child_weight: float = 1.0,
        xgb_tree_method: str = "hist",
        random_state: int | None = None,
    ):
        self.task_type = task_type
        self.n_estimators = int(n_estimators)
        self.n_inner_bootstraps = int(n_inner_bootstraps)
        self.eta = float(eta)
        self.instability_penalty = float(instability_penalty)
        self.weight_power = float(weight_power)
        self.weight_eps = float(weight_eps)
        self.update_target_mode = str(update_target_mode)
        self.transport_curvature_eps = float(transport_curvature_eps)
        self.denom_eps = float(denom_eps)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.weak_learner_backend = str(weak_learner_backend)
        self.xgb_learning_rate = float(xgb_learning_rate)
        self.xgb_subsample = float(xgb_subsample)
        self.xgb_colsample_bytree = float(xgb_colsample_bytree)
        self.xgb_reg_lambda = float(xgb_reg_lambda)
        self.xgb_min_child_weight = float(xgb_min_child_weight)
        self.xgb_tree_method = str(xgb_tree_method)
        self.random_state = random_state

    @staticmethod
    def _sigmoid(score: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(score, dtype=float), -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _loss_geometry(self, y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.task_type == "regression":
            first_order = np.asarray(y, dtype=float) - np.asarray(score, dtype=float)
            curvature = np.ones_like(first_order, dtype=float)
            return first_order, curvature
        proba = self._sigmoid(score)
        first_order = np.asarray(y, dtype=float) - proba
        curvature = proba * (1.0 - proba)
        return first_order, curvature

    def _fit_target_and_weight(
        self,
        first_order: np.ndarray,
        curvature: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self.update_target_mode == "legacy":
            return np.asarray(first_order, dtype=float), None
        if self.update_target_mode == "loss_aware":
            adjusted_curvature = np.asarray(curvature, dtype=float) + self.transport_curvature_eps
            target = np.asarray(first_order, dtype=float) / adjusted_curvature
            return target, adjusted_curvature
        raise ValueError(f"Unsupported update_target_mode={self.update_target_mode!r}")

    def _sampling_weights(self, pseudo_response: np.ndarray) -> np.ndarray:
        weights = np.power(np.abs(np.asarray(pseudo_response, dtype=float)) + self.weight_eps, self.weight_power)
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return np.full_like(weights, fill_value=1.0 / float(weights.size), dtype=float)
        return weights / weight_sum

    def _make_weak_learner(self, random_state: int | None):
        if self.weak_learner_backend == "sklearn_tree":
            return DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=random_state,
            )
        if self.weak_learner_backend == "xgb_tree":
            if XGBRegressor is None:  # pragma: no cover
                raise ImportError("xgboost is not installed, but weak_learner_backend='xgb_tree' was requested")
            if self.max_depth is None:
                raise ValueError("weak_learner_backend='xgb_tree' requires max_depth to be a positive integer")
            return XGBRegressor(
                n_estimators=1,
                max_depth=int(self.max_depth),
                learning_rate=self.xgb_learning_rate,
                subsample=self.xgb_subsample,
                colsample_bytree=self.xgb_colsample_bytree,
                reg_lambda=self.xgb_reg_lambda,
                min_child_weight=self.xgb_min_child_weight,
                objective="reg:squarederror",
                tree_method=self.xgb_tree_method,
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            )
        raise ValueError(f"Unsupported weak_learner_backend={self.weak_learner_backend!r}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConsensusTransportBoosting":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape={X.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Mismatched X/y with shapes {X.shape} and {y.shape}")
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task_type={self.task_type!r}")
        if self.update_target_mode not in {"legacy", "loss_aware"}:
            raise ValueError(f"Unsupported update_target_mode={self.update_target_mode!r}")
        if self.weak_learner_backend not in {"sklearn_tree", "xgb_tree"}:
            raise ValueError(f"Unsupported weak_learner_backend={self.weak_learner_backend!r}")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.n_inner_bootstraps <= 0:
            raise ValueError("n_inner_bootstraps must be positive")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if self.transport_curvature_eps < 0.0:
            raise ValueError("transport_curvature_eps must be non-negative")

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        train_score = np.zeros(n_samples, dtype=float)

        self.learners_ = []
        self.alphas_ = []
        self.train_score_path_ = []
        self.n_features_in_ = X.shape[1]

        for _ in range(self.n_estimators):
            first_order, curvature = self._loss_geometry(y, train_score)
            fit_target, fit_sample_weight = self._fit_target_and_weight(first_order, curvature)
            sampling_weights = self._sampling_weights(first_order)

            round_learners = []
            round_predictions = np.empty((self.n_inner_bootstraps, n_samples), dtype=float)
            for bootstrap_idx in range(self.n_inner_bootstraps):
                sample_idx = rng.choice(n_samples, size=n_samples, replace=True, p=sampling_weights)
                learner = self._make_weak_learner(random_state=int(rng.integers(0, 2**31 - 1)))
                learner.fit(
                    X[sample_idx],
                    fit_target[sample_idx],
                    sample_weight=None if fit_sample_weight is None else fit_sample_weight[sample_idx],
                )
                round_learners.append(learner)
                round_predictions[bootstrap_idx] = np.asarray(learner.predict(X), dtype=float).reshape(-1)

            consensus = round_predictions.mean(axis=0)
            instability = np.mean((round_predictions - consensus[None, :]) ** 2, axis=0)

            numerator = float(np.dot(first_order, consensus))
            denom_main = (1.0 / self.eta) * float(np.dot(consensus, consensus))
            denom_penalty = 2.0 * self.instability_penalty * float(np.dot(instability, consensus**2))
            denominator = max(denom_main + denom_penalty, self.denom_eps)
            alpha = numerator / denominator

            train_score = train_score + alpha * consensus
            self.learners_.append(round_learners)
            self.alphas_.append(float(alpha))
            self.train_score_path_.append(train_score.copy())

        self.alphas_ = np.asarray(self.alphas_, dtype=float)
        self.train_score_ = train_score
        return self

    def _round_consensus_prediction(self, X: np.ndarray, round_learners: list[object]) -> np.ndarray:
        return np.mean(
            [np.asarray(learner.predict(X), dtype=float).reshape(-1) for learner in round_learners],
            axis=0,
        )

    def decision_function_staged(self, X: np.ndarray, checkpoints: list[int] | np.ndarray | None = None) -> dict[int, np.ndarray]:
        if not hasattr(self, "learners_"):
            raise ValueError("Model is not fitted")
        X = np.asarray(X, dtype=float)
        if checkpoints is None:
            requested = list(range(1, len(self.learners_) + 1))
        else:
            requested = sorted({int(x) for x in checkpoints if int(x) > 0})
        if not requested:
            return {}
        max_requested = max(requested)
        if max_requested > len(self.learners_):
            raise ValueError(
                f"Requested checkpoint {max_requested}, but model only has {len(self.learners_)} boosting rounds"
            )
        out: dict[int, np.ndarray] = {}
        requested_set = set(requested)
        score = np.zeros(X.shape[0], dtype=float)
        for idx, (alpha, round_learners) in enumerate(zip(self.alphas_, self.learners_), start=1):
            score = score + float(alpha) * self._round_consensus_prediction(X, round_learners)
            if idx in requested_set:
                out[idx] = score.copy()
            if len(out) == len(requested):
                break
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        staged = self.decision_function_staged(X, checkpoints=[len(self.learners_)])
        return staged[len(self.learners_)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = self.decision_function(X)
        if self.task_type == "classification":
            return (self._sigmoid(score) >= 0.5).astype(float)
        return score

    def predict_proba_staged(self, X: np.ndarray, checkpoints: list[int] | np.ndarray | None = None) -> dict[int, np.ndarray]:
        if self.task_type != "classification":
            raise ValueError("predict_proba_staged is only available for classification models")
        staged_scores = self.decision_function_staged(X, checkpoints=checkpoints)
        out: dict[int, np.ndarray] = {}
        for checkpoint, score in staged_scores.items():
            proba_one = self._sigmoid(score)
            out[int(checkpoint)] = np.column_stack([1.0 - proba_one, proba_one])
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification models")
        staged = self.predict_proba_staged(X, checkpoints=[len(self.learners_)])
        return staged[len(self.learners_)]
