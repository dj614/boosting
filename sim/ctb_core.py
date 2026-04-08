from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor


TaskType = Literal["regression", "classification"]


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
        transport_curvature_eps: float = 1e-6,
        denom_eps: float = 1e-12,
        max_depth: int | None = 1,
        max_leaf_nodes: int | None = 10,
        min_samples_leaf: int = 5,
        leaf_ridge: float = 1.0,
        random_state: int | None = None,
    ):
        self.task_type = task_type
        self.n_estimators = int(n_estimators)
        self.n_inner_bootstraps = int(n_inner_bootstraps)
        self.eta = float(eta)
        self.instability_penalty = float(instability_penalty)
        self.weight_power = float(weight_power)
        self.weight_eps = float(weight_eps)
        self.transport_curvature_eps = float(transport_curvature_eps)
        self.denom_eps = float(denom_eps)
        self.max_depth = max_depth
        self.max_leaf_nodes = None if max_leaf_nodes is None else int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.leaf_ridge = float(leaf_ridge)
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
        if np.any(positions < 0) or np.any(positions >= unique_leaf_ids.shape[0]):
            raise RuntimeError("Encountered leaf ids outside the learned CTB partition")
        if not np.array_equal(unique_leaf_ids[positions], applied_leaf_ids):
            raise RuntimeError("Encountered unknown leaf ids while mapping CTB leaf values")
        return np.asarray(positions, dtype=int)

    def _validate_round_state(self, round_state: dict[str, Any], *, expected_bootstraps: int) -> None:
        required_keys = {
            "tree",
            "leaf_ids",
            "leaf_values",
            "leaf_instability",
            "leaf_mass",
            "bootstrap_leaf_values",
        }
        missing = required_keys.difference(round_state)
        if missing:
            raise RuntimeError(f"Malformed CTB round state; missing keys: {sorted(missing)}")
        leaf_ids = np.asarray(round_state["leaf_ids"], dtype=np.int64).reshape(-1)
        leaf_values = np.asarray(round_state["leaf_values"], dtype=float).reshape(-1)
        leaf_instability = np.asarray(round_state["leaf_instability"], dtype=float).reshape(-1)
        leaf_mass = np.asarray(round_state["leaf_mass"], dtype=float).reshape(-1)
        bootstrap_leaf_values = np.asarray(round_state["bootstrap_leaf_values"], dtype=float)
        n_leaves = int(leaf_ids.shape[0])
        if n_leaves <= 0:
            raise RuntimeError("CTB round state must contain at least one leaf")
        if leaf_values.shape[0] != n_leaves or leaf_instability.shape[0] != n_leaves or leaf_mass.shape[0] != n_leaves:
            raise RuntimeError("CTB leaf summaries must align with the learned partition")
        if bootstrap_leaf_values.shape != (expected_bootstraps, n_leaves):
            raise RuntimeError(
                "Bootstrap leaf values must have shape (n_inner_bootstraps, n_leaves); "
                f"got {bootstrap_leaf_values.shape}"
            )
        if not np.array_equal(np.sort(leaf_ids), leaf_ids):
            raise RuntimeError("CTB leaf ids must be stored in sorted order")
        if np.any(leaf_mass < 0.0):
            raise RuntimeError("CTB leaf masses must be non-negative")
        if np.any(leaf_instability < -1e-12):
            raise RuntimeError("CTB leaf instability must be non-negative")
        bootstrap_mean = np.mean(bootstrap_leaf_values, axis=0)
        bootstrap_var = np.mean((bootstrap_leaf_values - bootstrap_mean[None, :]) ** 2, axis=0)
        if not np.allclose(leaf_values, bootstrap_mean, atol=1e-10, rtol=1e-10):
            raise RuntimeError("Stored CTB consensus leaf values do not match bootstrap means")
        if not np.allclose(leaf_instability, bootstrap_var, atol=1e-10, rtol=1e-10):
            raise RuntimeError("Stored CTB leaf instability does not match bootstrap variance")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConsensusTransportBoosting":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D X, got shape={X.shape}")
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Mismatched X/y with shapes {X.shape} and {y.shape}")
        if self.task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported task_type={self.task_type!r}")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.n_inner_bootstraps <= 0:
            raise ValueError("n_inner_bootstraps must be positive")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if self.instability_penalty < 0.0:
            raise ValueError("instability_penalty must be non-negative")
        if self.weight_power < 0.0:
            raise ValueError("weight_power must be non-negative")
        if self.transport_curvature_eps < 0.0:
            raise ValueError("transport_curvature_eps must be non-negative")
        if self.weight_eps < 0.0:
            raise ValueError("weight_eps must be non-negative")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        if self.max_leaf_nodes is not None and self.max_leaf_nodes <= 1:
            raise ValueError("max_leaf_nodes must be greater than 1 when specified")
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
            round_state = {
                "tree": tree,
                "leaf_ids": unique_leaf_ids,
                "leaf_values": consensus_leaf_values,
                "leaf_instability": leaf_instability,
                "leaf_mass": leaf_mass,
                "bootstrap_leaf_values": leaf_values_boot,
            }
            self._validate_round_state(round_state, expected_bootstraps=self.n_inner_bootstraps)
            self.round_states_.append(round_state)
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
