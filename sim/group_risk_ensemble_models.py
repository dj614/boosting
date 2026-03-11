from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

from .ctb_core import ConsensusTransportBoosting
from .grouped_classification_data import BinaryClassificationSplit

try:  # pragma: no cover
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


Array = np.ndarray


@dataclass(frozen=True)
class EnsembleModelConfig:
    family: str
    max_depth: int
    n_estimators: int
    min_samples_leaf: int = 5
    learning_rate: float = 0.05
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    inner_bootstraps: int = 8
    eta: float = 1.0
    instability_penalty: float = 0.0
    weight_power: float = 1.0
    weight_eps: float = 1e-8
    random_state: int = 0

    @property
    def model_name(self) -> str:
        return f"{self.family}_depth{self.max_depth}"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class BinaryEnsembleWrapper:
    def __init__(
        self,
        config: EnsembleModelConfig,
        selection_checkpoints: Sequence[int],
        trajectory_checkpoints: Sequence[int],
    ) -> None:
        self.config = config
        self.selection_checkpoints = sorted({int(x) for x in selection_checkpoints if int(x) > 0})
        self.trajectory_checkpoints = sorted({int(x) for x in trajectory_checkpoints if int(x) > 0})
        if not self.selection_checkpoints:
            raise ValueError("selection_checkpoints must be non-empty")
        if not self.trajectory_checkpoints:
            raise ValueError("trajectory_checkpoints must be non-empty")
        self.model = None
        self.selected_checkpoint_: Optional[int] = None
        self.selection_trace_: Optional[pd.DataFrame] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(
        self,
        train_split: BinaryClassificationSplit,
        valid_split: BinaryClassificationSplit,
    ) -> "BinaryEnsembleWrapper":
        self.model = self._build_estimator()
        self.model.fit(train_split.X, train_split.y)
        valid_probs = self.predict_proba_staged(valid_split.X, checkpoints=self.selection_checkpoints)
        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_loss = None
        for checkpoint in self.selection_checkpoints:
            prob = np.asarray(valid_probs[int(checkpoint)], dtype=float)
            current_loss = float(log_loss(valid_split.y, np.clip(prob, 1e-8, 1.0 - 1e-8), labels=[0, 1]))
            rows.append({"checkpoint": int(checkpoint), "valid_log_loss": current_loss})
            if best_loss is None or current_loss < best_loss - 1e-12 or (
                abs(current_loss - best_loss) <= 1e-12 and int(checkpoint) < int(best_checkpoint)
            ):
                best_loss = current_loss
                best_checkpoint = int(checkpoint)
        self.selected_checkpoint_ = int(best_checkpoint)
        self.selection_trace_ = pd.DataFrame(rows)
        return self

    def predict_proba(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        return np.asarray(self.predict_proba_staged(X, checkpoints=[use_checkpoint])[use_checkpoint], dtype=float)

    def predict(self, X: Array, checkpoint: Optional[int] = None, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X, checkpoint=checkpoint) >= threshold).astype(int)

    def predict_proba_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        return self._predict_proba_at_checkpoints(X, checkpoints=checkpoints)

    def trajectory(self, X: Array) -> Dict[int, Array]:
        return self.predict_proba_staged(X, checkpoints=self.trajectory_checkpoints)

    def _build_estimator(self):
        raise NotImplementedError

    def _predict_proba_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        raise NotImplementedError


class BaggingBinaryWrapper(BinaryEnsembleWrapper):
    def _build_estimator(self):
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
            ),
            n_estimators=self.config.n_estimators,
            bootstrap=True,
            random_state=self.config.random_state,
            n_jobs=1,
        )

    def _predict_proba_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        estimator_probs = [np.asarray(est.predict_proba(X)[:, 1], dtype=float) for est in self.model.estimators_]
        return _prefix_average_predictions(estimator_probs=estimator_probs, checkpoints=checkpoints)


class RandomForestBinaryWrapper(BinaryEnsembleWrapper):
    def _build_estimator(self):
        return RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            bootstrap=True,
            random_state=self.config.random_state,
            n_jobs=1,
        )

    def _predict_proba_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        estimator_probs = [np.asarray(est.predict_proba(X)[:, 1], dtype=float) for est in self.model.estimators_]
        return _prefix_average_predictions(estimator_probs=estimator_probs, checkpoints=checkpoints)


class GradientBoostingBinaryWrapper(BinaryEnsembleWrapper):
    def _build_estimator(self):
        return GradientBoostingClassifier(
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )

    def _predict_proba_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        requested = sorted({int(x) for x in checkpoints})
        requested_set = set(requested)
        out: Dict[int, Array] = {}
        for idx, proba in enumerate(self.model.staged_predict_proba(X), start=1):
            if idx in requested_set:
                out[idx] = np.asarray(proba[:, 1], dtype=float)
            if len(out) == len(requested):
                break
        _validate_checkpoint_outputs(requested, out, self.config.n_estimators)
        return out


class XGBoostBinaryWrapper(BinaryEnsembleWrapper):
    def _build_estimator(self):
        if XGBClassifier is None:  # pragma: no cover
            raise ImportError("xgboost is not installed, but family='xgb' was requested")
        return XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=1.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0,
        )

    def _predict_proba_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        requested = sorted({int(x) for x in checkpoints})
        out: Dict[int, Array] = {}
        for checkpoint in requested:
            proba = self.model.predict_proba(X, iteration_range=(0, int(checkpoint)))
            out[int(checkpoint)] = np.asarray(proba[:, 1], dtype=float)
        return out


class CTBBinaryWrapper(BinaryEnsembleWrapper):
    def _build_estimator(self):
        return ConsensusTransportBoosting(
            task_type="classification",
            n_estimators=self.config.n_estimators,
            n_inner_bootstraps=self.config.inner_bootstraps,
            eta=self.config.eta,
            instability_penalty=self.config.instability_penalty,
            weight_power=self.config.weight_power,
            weight_eps=self.config.weight_eps,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )

    def _predict_proba_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        requested = sorted({int(x) for x in checkpoints})
        staged = self.model.predict_proba_staged(X, checkpoints=requested)
        out: Dict[int, Array] = {}
        for checkpoint, proba in staged.items():
            out[int(checkpoint)] = np.asarray(proba[:, 1], dtype=float)
        _validate_checkpoint_outputs(requested, out, self.config.n_estimators)
        return out

def _validate_checkpoint_outputs(requested: Sequence[int], out: Mapping[int, Array], max_checkpoint: int) -> None:
    missing = [int(x) for x in requested if int(x) not in out]
    if missing:
        raise RuntimeError(
            f"Failed to produce staged predictions for checkpoints {missing}; max_checkpoint={max_checkpoint}"
        )



def _prefix_average_predictions(estimator_probs: Sequence[Array], checkpoints: Sequence[int]) -> Dict[int, Array]:
    requested = sorted({int(x) for x in checkpoints})
    if not requested:
        return {}
    max_requested = max(requested)
    if max_requested > len(estimator_probs):
        raise ValueError(
            f"Requested checkpoint {max_requested}, but only {len(estimator_probs)} estimators are available"
        )

    running = np.zeros_like(estimator_probs[0], dtype=float)
    out: Dict[int, Array] = {}
    requested_set = set(requested)
    for idx, prob in enumerate(estimator_probs, start=1):
        running += prob
        if idx in requested_set:
            out[idx] = running / float(idx)
    _validate_checkpoint_outputs(requested, out, len(estimator_probs))
    return out



def build_binary_ensemble_wrapper(
    config: EnsembleModelConfig,
    selection_checkpoints: Sequence[int],
    trajectory_checkpoints: Sequence[int],
) -> BinaryEnsembleWrapper:
    family = str(config.family)
    if family == "bagging":
        return BaggingBinaryWrapper(config, selection_checkpoints, trajectory_checkpoints)
    if family == "rf":
        return RandomForestBinaryWrapper(config, selection_checkpoints, trajectory_checkpoints)
    if family == "gbdt":
        return GradientBoostingBinaryWrapper(config, selection_checkpoints, trajectory_checkpoints)
    if family == "xgb":
        return XGBoostBinaryWrapper(config, selection_checkpoints, trajectory_checkpoints)
    if family == "ctb":
        return CTBBinaryWrapper(config, selection_checkpoints, trajectory_checkpoints)
    raise ValueError(f"Unsupported family={family!r}")



def expand_model_grid(
    families: Sequence[str],
    max_depths: Sequence[int],
    n_estimators: int,
    learning_rate: float,
    min_samples_leaf: int,
    subsample: float,
    colsample_bytree: float,
    inner_bootstraps: int,
    eta: float,
    instability_penalty: float,
    weight_power: float,
    weight_eps: float,
    random_state: int,
) -> List[EnsembleModelConfig]:
    grid: List[EnsembleModelConfig] = []
    for family in families:
        for depth in max_depths:
            grid.append(
                EnsembleModelConfig(
                    family=str(family),
                    max_depth=int(depth),
                    n_estimators=int(n_estimators),
                    min_samples_leaf=int(min_samples_leaf),
                    learning_rate=float(learning_rate),
                    subsample=float(subsample),
                    colsample_bytree=float(colsample_bytree),
                    inner_bootstraps=int(inner_bootstraps),
                    eta=float(eta),
                    instability_penalty=float(instability_penalty),
                    weight_power=float(weight_power),
                    weight_eps=float(weight_eps),
                    random_state=int(random_state),
                )
            )
    return grid
