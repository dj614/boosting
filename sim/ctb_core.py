from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor


TaskType = Literal["regression", "classification"]
UpdateTargetMode = Literal["legacy", "loss_aware"]
WeakLearnerBackend = Literal["sklearn_tree", "xgb_tree"]


class ConsensusTransportBoosting(BaseEstimator):
    """Minimal CTB implementation for regression and binary classification.

    This implementation follows the shared-structure leaf-bootstrap CTB update:
    each round learns one tree partition from loss-aware pseudo-targets and
    hard-sample weights, then applies multiplier bootstrap only to the leaf
    values while keeping the tree structure fixed.
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
        update_target_mode: UpdateTargetMode = "loss_aware",
        transport_curvature_eps: float = 1e-6,
        denom_eps: float = 1e-12,
        max_depth: int | None = 1,
        max_leaf_nodes: int | None = 10,
        min_samples_leaf: int = 5,
        weak_learner_backend: WeakLearnerBackend = "sklearn_tree",
        leaf_ridge: float | None = None,
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
        self.max_leaf_nodes = None if max_leaf_nodes is None else int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.weak_learner_backend = str(weak_learner_backend)
        self.leaf_ridge = float(xgb_reg_lambda if leaf_ridge is None else leaf_ridge)
        # Kept for step-1 backward compatibility with existing wrappers/configs.
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

    def _initial_constant_score(self, y: np.ndarray) -> float:
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if self.task_type == "regression":
            return float(np.mean(y_arr))
        positive_rate = float(np.clip(np.mean(y_arr), self.denom_eps, 1.0 - self.denom_eps))
        return float(np.log(positive_rate / (1.0 - positive_rate)))

    def _loss_geometry(self, y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        score_arr = np.asarray(score, dtype=float).reshape(-1)
        if self.task_type == "regression":
            grad = score_arr - y_arr
            hess = np.ones_like(grad, dtype=float)
        else:
            proba = self._sigmoid(score_arr)
            grad = proba - y_arr
            hess = proba * (1.0 - proba)
        neg_grad = -grad
        z = neg_grad / (hess + self.transport_curvature_eps)
        s = np.power(np.abs(grad) + self.weight_eps, self.weight_power)
        return grad, hess, z, s

    def _make_structure_learner(self, random_state: int | None):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state,
        )

    @staticmethod
    def _leaf_lookup(unique_leaf_ids: np.ndarray, applied_leaf_ids: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(unique_leaf_ids, applied_leaf_ids)
        return np.asarray(positions, dtype=int)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConsensusTransportBoosting":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape={X.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Mismatched X/y with shapes {X.shape} and {y.shape}")
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task_type={self.task_type!r}")
        if self.update_target_mode != "loss_aware":
            raise ValueError(
                "Step-1 CTB core only supports update_target_mode='loss_aware'; "
                f"got {self.update_target_mode!r}"
            )
        if self.weak_learner_backend != "sklearn_tree":
            raise ValueError(
                "Step-1 CTB core only supports weak_learner_backend='sklearn_tree'; "
                f"got {self.weak_learner_backend!r}"
            )
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.n_inner_bootstraps <= 0:
            raise ValueError("n_inner_bootstraps must be positive")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if self.transport_curvature_eps < 0.0:
            raise ValueError("transport_curvature_eps must be non-negative")
        if self.weight_eps < 0.0:
            raise ValueError("weight_eps must be non-negative")
        if self.leaf_ridge < 0.0:
            raise ValueError("leaf_ridge must be non-negative")

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        init_score = self._initial_constant_score(y)
        train_score = np.full(n_samples, fill_value=init_score, dtype=float)

        self.learners_: list[list[Any]] = []
        self.round_states_: list[dict[str, Any]] = []
        self.alphas_ = []
        self.n_features_in_ = X.shape[1]
        self.init_score_ = float(init_score)

        for _ in range(self.n_estimators):
            grad, hess, z, s = self._loss_geometry(y, train_score)
            tree = self._make_structure_learner(random_state=int(rng.integers(0, 2**31 - 1)))
            tree.fit(X, z, sample_weight=s)

            applied_leaf_ids = np.asarray(tree.apply(X), dtype=np.int64)
            unique_leaf_ids, inverse = np.unique(applied_leaf_ids, return_inverse=True)
            n_leaves = int(unique_leaf_ids.size)
            if n_leaves <= 0:
                raise RuntimeError("Tree learner produced no leaves")

            leaf_values_boot = np.empty((self.n_inner_bootstraps, n_leaves), dtype=float)
            for bootstrap_idx in range(self.n_inner_bootstraps):
                multiplier = rng.poisson(1.0, size=n_samples).astype(float)
                weighted = multiplier * s
                numerator = np.bincount(inverse, weights=weighted * z, minlength=n_leaves)
                denominator = np.bincount(inverse, weights=weighted, minlength=n_leaves) + self.leaf_ridge
                leaf_values_boot[bootstrap_idx] = numerator / np.maximum(denominator, self.denom_eps)

            consensus_leaf_values = np.mean(leaf_values_boot, axis=0)
            leaf_instability = np.mean((leaf_values_boot - consensus_leaf_values[None, :]) ** 2, axis=0)
            leaf_mass = np.bincount(inverse, weights=s, minlength=n_leaves).astype(float)
            consensus = np.asarray(consensus_leaf_values[inverse], dtype=float)

            numerator = float(np.dot(-grad, consensus))
            denom_hess = float(np.dot(hess, consensus**2))
            denom_step = (1.0 / self.eta) * float(np.dot(consensus, consensus))
            denom_penalty = 2.0 * self.instability_penalty * float(
                np.dot(leaf_instability * leaf_mass, consensus_leaf_values**2)
            )
            denominator = denom_hess + denom_step + denom_penalty
            alpha = max(0.0, numerator / (denominator + self.denom_eps))

            train_score = train_score + alpha * consensus
            self.learners_.append([tree])
            self.round_states_.append(
                {
                    "tree": tree,
                    "leaf_ids": unique_leaf_ids,
                    "leaf_values": consensus_leaf_values,
                    "leaf_instability": leaf_instability,
                    "leaf_mass": leaf_mass,
                }
            )
            self.alphas_.append(float(alpha))

        self.alphas_ = np.asarray(self.alphas_, dtype=float)
        self.train_score_ = train_score
        return self

    def _round_consensus_prediction(self, X: np.ndarray, round_state: dict[str, Any]) -> np.ndarray:
        tree = round_state["tree"]
        unique_leaf_ids = np.asarray(round_state["leaf_ids"], dtype=np.int64)
        leaf_values = np.asarray(round_state["leaf_values"], dtype=float)
        applied_leaf_ids = np.asarray(tree.apply(np.asarray(X, dtype=float)), dtype=np.int64)
        positions = self._leaf_lookup(unique_leaf_ids, applied_leaf_ids)
        return np.asarray(leaf_values[positions], dtype=float).reshape(-1)

    def decision_function_staged(self, X: np.ndarray, checkpoints: list[int] | np.ndarray | None = None) -> dict[int, np.ndarray]:
        if not hasattr(self, "round_states_"):
            raise ValueError("Model is not fitted")
        X = np.asarray(X, dtype=float)
        if checkpoints is None:
            requested = list(range(1, len(self.round_states_) + 1))
        else:
            requested = sorted({int(x) for x in checkpoints if int(x) > 0})
        if not requested:
            return {}
        max_requested = max(requested)
        if max_requested > len(self.round_states_):
            raise ValueError(
                f"Requested checkpoint {max_requested}, but model only has {len(self.round_states_)} boosting rounds"
            )
        out: dict[int, np.ndarray] = {}
        requested_set = set(requested)
        score = np.full(X.shape[0], fill_value=self.init_score_, dtype=float)
        for idx, (alpha, round_state) in enumerate(zip(self.alphas_, self.round_states_), start=1):
            score = score + float(alpha) * self._round_consensus_prediction(X, round_state)
            if idx in requested_set:
                out[idx] = score.copy()
            if len(out) == len(requested):
                break
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        staged = self.decision_function_staged(X, checkpoints=[len(self.round_states_)])
        return staged[len(self.round_states_)]

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
        staged = self.predict_proba_staged(X, checkpoints=[len(self.round_states_)])
        return staged[len(self.round_states_)]
