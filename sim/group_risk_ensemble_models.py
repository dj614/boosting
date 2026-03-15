from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .ctb_core import ConsensusTransportBoosting
from .ctb_semantics import ctb_family_output_name, ctb_tree_model_name, is_ctb_tree_family_name, normalize_ctb_tree_family_name

try:  # pragma: no cover
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


Array = np.ndarray


@dataclass(frozen=True)
class EnsembleModelConfig:
    family: str
    max_depth: int
    n_estimators: int
    task_type: str = "classification"
    min_samples_leaf: int = 5
    learning_rate: float = 0.05
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    inner_bootstraps: int = 8
    eta: float = 1.0
    instability_penalty: float = 0.0
    weight_power: float = 1.0
    weight_eps: float = 1e-8
    ctb_target_mode: str = "loss_aware"
    ctb_curvature_eps: float = 1e-6
    ctb_weak_learner_backend: str = "sklearn_tree"
    ctb_xgb_reg_lambda: float = 1.0
    ctb_xgb_min_child_weight: float = 1.0
    random_state: int = 0

    @property
    def model_name(self) -> str:
        family = normalize_ctb_tree_family_name(self.family)
        if is_ctb_tree_family_name(family):
            return ctb_tree_model_name(
                depth=int(self.max_depth),
                task_type=str(self.task_type),
                update_target_mode=str(self.ctb_target_mode),
                transport_curvature_eps=float(self.ctb_curvature_eps),
                weak_learner_backend=str(self.ctb_weak_learner_backend),
                include_task_suffix=True,
            )
        base = f"{family}_depth{self.max_depth}"
        if str(self.task_type).strip().lower() != "classification":
            base = f"{base}_{str(self.task_type).strip().lower()}"
        return base

    @property
    def family_output_name(self) -> str:
        return ctb_family_output_name(
            family_name=self.family,
            weak_learner_backend=self.ctb_weak_learner_backend,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class EnsembleWrapperBase:
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

    @property
    def family_name(self) -> str:
        return self.config.family_output_name

    @property
    def selection_metric_name(self) -> str:
        raise NotImplementedError

    def _selection_metric_value(self, y_true: Array, prediction: Array) -> float:
        raise NotImplementedError

    def fit(self, train_split, valid_split):
        self.model = self._build_estimator()
        self.model.fit(train_split.X, train_split.y)
        valid_predictions = self._predict_at_checkpoints(valid_split.X, checkpoints=self.selection_checkpoints)
        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_metric = None
        for checkpoint in self.selection_checkpoints:
            pred = np.asarray(valid_predictions[int(checkpoint)], dtype=float)
            current_metric = float(self._selection_metric_value(valid_split.y, pred))
            rows.append({"checkpoint": int(checkpoint), self.selection_metric_name: current_metric})
            if best_metric is None or current_metric < best_metric - 1e-12 or (
                abs(current_metric - best_metric) <= 1e-12 and int(checkpoint) < int(best_checkpoint)
            ):
                best_metric = current_metric
                best_checkpoint = int(checkpoint)
        self.selected_checkpoint_ = int(best_checkpoint)
        self.selection_trace_ = pd.DataFrame(rows)
        return self

    def _build_estimator(self):
        raise NotImplementedError

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        raise NotImplementedError


class BinaryEnsembleWrapper(EnsembleWrapperBase):
    @property
    def selection_metric_name(self) -> str:
        return "valid_log_loss"

    def _selection_metric_value(self, y_true: Array, prediction: Array) -> float:
        prob = np.clip(np.asarray(prediction, dtype=float), 1e-8, 1.0 - 1e-8)
        return float(log_loss(np.asarray(y_true, dtype=int), prob, labels=[0, 1]))

    def predict_proba(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        return np.asarray(self._predict_at_checkpoints(X, checkpoints=[use_checkpoint])[use_checkpoint], dtype=float)

    def predict(self, X: Array, checkpoint: Optional[int] = None, threshold: float = 0.5) -> Array:
        return (self.predict_proba(X, checkpoint=checkpoint) >= float(threshold)).astype(int)

    def predict_proba_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        return self._predict_at_checkpoints(X, checkpoints=checkpoints)

    def trajectory(self, X: Array) -> Dict[int, Array]:
        return self.predict_proba_staged(X, checkpoints=self.trajectory_checkpoints)


class RegressionEnsembleWrapper(EnsembleWrapperBase):
    @property
    def selection_metric_name(self) -> str:
        return "valid_mse"

    def _selection_metric_value(self, y_true: Array, prediction: Array) -> float:
        return float(mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(prediction, dtype=float)))

    def predict(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        return np.asarray(self._predict_at_checkpoints(X, checkpoints=[use_checkpoint])[use_checkpoint], dtype=float)

    def predict_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        return self._predict_at_checkpoints(X, checkpoints=checkpoints)

    def trajectory(self, X: Array) -> Dict[int, Array]:
        return self.predict_staged(X, checkpoints=self.trajectory_checkpoints)


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

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        estimator_probs = [np.asarray(est.predict_proba(X)[:, 1], dtype=float) for est in self.model.estimators_]
        return _prefix_average_predictions(estimator_outputs=estimator_probs, checkpoints=checkpoints)


class BaggingRegressionWrapper(RegressionEnsembleWrapper):
    def _build_estimator(self):
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
            ),
            n_estimators=self.config.n_estimators,
            bootstrap=True,
            random_state=self.config.random_state,
            n_jobs=1,
        )

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        estimator_preds = [np.asarray(est.predict(X), dtype=float) for est in self.model.estimators_]
        return _prefix_average_predictions(estimator_outputs=estimator_preds, checkpoints=checkpoints)


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

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        estimator_probs = [np.asarray(est.predict_proba(X)[:, 1], dtype=float) for est in self.model.estimators_]
        return _prefix_average_predictions(estimator_outputs=estimator_probs, checkpoints=checkpoints)


class RandomForestRegressionWrapper(RegressionEnsembleWrapper):
    def _build_estimator(self):
        return RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            bootstrap=True,
            random_state=self.config.random_state,
            n_jobs=1,
        )

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        estimator_preds = [np.asarray(est.predict(X), dtype=float) for est in self.model.estimators_]
        return _prefix_average_predictions(estimator_outputs=estimator_preds, checkpoints=checkpoints)


class GradientBoostingBinaryWrapper(BinaryEnsembleWrapper):
    def _build_estimator(self):
        return GradientBoostingClassifier(
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
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


class GradientBoostingRegressionWrapper(RegressionEnsembleWrapper):
    def _build_estimator(self):
        return GradientBoostingRegressor(
            learning_rate=self.config.learning_rate,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            random_state=self.config.random_state,
        )

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        requested = sorted({int(x) for x in checkpoints})
        requested_set = set(requested)
        out: Dict[int, Array] = {}
        for idx, pred in enumerate(self.model.staged_predict(X), start=1):
            if idx in requested_set:
                out[idx] = np.asarray(pred, dtype=float)
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

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        requested = sorted({int(x) for x in checkpoints})
        out: Dict[int, Array] = {}
        for checkpoint in requested:
            proba = self.model.predict_proba(X, iteration_range=(0, int(checkpoint)))
            out[int(checkpoint)] = np.asarray(proba[:, 1], dtype=float)
        return out


class XGBoostRegressionWrapper(RegressionEnsembleWrapper):
    def _build_estimator(self):
        if XGBRegressor is None:  # pragma: no cover
            raise ImportError("xgboost is not installed, but family='xgb' was requested")
        return XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=self.config.random_state,
            n_jobs=1,
            verbosity=0,
        )

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        requested = sorted({int(x) for x in checkpoints})
        out: Dict[int, Array] = {}
        for checkpoint in requested:
            pred = self.model.predict(X, iteration_range=(0, int(checkpoint)))
            out[int(checkpoint)] = np.asarray(pred, dtype=float)
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
            update_target_mode=self.config.ctb_target_mode,
            transport_curvature_eps=self.config.ctb_curvature_eps,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            weak_learner_backend=self.config.ctb_weak_learner_backend,
            xgb_learning_rate=self.config.learning_rate,
            xgb_subsample=self.config.subsample,
            xgb_colsample_bytree=self.config.colsample_bytree,
            xgb_reg_lambda=self.config.ctb_xgb_reg_lambda,
            xgb_min_child_weight=self.config.ctb_xgb_min_child_weight,
            random_state=self.config.random_state,
        )

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        requested = sorted({int(x) for x in checkpoints})
        staged = self.model.predict_proba_staged(X, checkpoints=requested)
        out: Dict[int, Array] = {}
        for checkpoint, proba in staged.items():
            out[int(checkpoint)] = np.asarray(proba[:, 1], dtype=float)
        _validate_checkpoint_outputs(requested, out, self.config.n_estimators)
        return out


class CTBRegressionWrapper(RegressionEnsembleWrapper):
    def _build_estimator(self):
        return ConsensusTransportBoosting(
            task_type="regression",
            n_estimators=self.config.n_estimators,
            n_inner_bootstraps=self.config.inner_bootstraps,
            eta=self.config.eta,
            instability_penalty=self.config.instability_penalty,
            weight_power=self.config.weight_power,
            weight_eps=self.config.weight_eps,
            update_target_mode=self.config.ctb_target_mode,
            transport_curvature_eps=self.config.ctb_curvature_eps,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            weak_learner_backend=self.config.ctb_weak_learner_backend,
            xgb_learning_rate=self.config.learning_rate,
            xgb_subsample=self.config.subsample,
            xgb_colsample_bytree=self.config.colsample_bytree,
            xgb_reg_lambda=self.config.ctb_xgb_reg_lambda,
            xgb_min_child_weight=self.config.ctb_xgb_min_child_weight,
            random_state=self.config.random_state,
        )

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        requested = sorted({int(x) for x in checkpoints})
        staged = self.model.decision_function_staged(X, checkpoints=requested)
        out: Dict[int, Array] = {}
        for checkpoint, pred in staged.items():
            out[int(checkpoint)] = np.asarray(pred, dtype=float)
        _validate_checkpoint_outputs(requested, out, self.config.n_estimators)
        return out


def _validate_checkpoint_outputs(requested: Sequence[int], out: Mapping[int, Array], max_checkpoint: int) -> None:
    missing = [int(x) for x in requested if int(x) not in out]
    if missing:
        raise RuntimeError(
            f"Failed to produce staged predictions for checkpoints {missing}; max_checkpoint={max_checkpoint}"
        )


def _prefix_average_predictions(estimator_outputs: Sequence[Array], checkpoints: Sequence[int]) -> Dict[int, Array]:
    requested = sorted({int(x) for x in checkpoints})
    if not requested:
        return {}
    max_requested = max(requested)
    if max_requested > len(estimator_outputs):
        raise ValueError(
            f"Requested checkpoint {max_requested}, but only {len(estimator_outputs)} estimators are available"
        )
    running = np.zeros_like(np.asarray(estimator_outputs[0], dtype=float), dtype=float)
    out: Dict[int, Array] = {}
    requested_set = set(requested)
    for idx, values in enumerate(estimator_outputs, start=1):
        running += np.asarray(values, dtype=float)
        if idx in requested_set:
            out[idx] = running / float(idx)
    _validate_checkpoint_outputs(requested, out, len(estimator_outputs))
    return out


def build_ensemble_wrapper(
    config: EnsembleModelConfig,
    selection_checkpoints: Sequence[int],
    trajectory_checkpoints: Sequence[int],
):
    family = normalize_ctb_tree_family_name(config.family)
    task_type = str(config.task_type).strip().lower()
    if task_type == "classification":
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
    if task_type == "regression":
        if family == "bagging":
            return BaggingRegressionWrapper(config, selection_checkpoints, trajectory_checkpoints)
        if family == "rf":
            return RandomForestRegressionWrapper(config, selection_checkpoints, trajectory_checkpoints)
        if family == "gbdt":
            return GradientBoostingRegressionWrapper(config, selection_checkpoints, trajectory_checkpoints)
        if family == "xgb":
            return XGBoostRegressionWrapper(config, selection_checkpoints, trajectory_checkpoints)
        if family == "ctb":
            return CTBRegressionWrapper(config, selection_checkpoints, trajectory_checkpoints)
    raise ValueError(f"Unsupported task_type={task_type!r} family={family!r}")


def build_binary_ensemble_wrapper(
    config: EnsembleModelConfig,
    selection_checkpoints: Sequence[int],
    trajectory_checkpoints: Sequence[int],
) -> BinaryEnsembleWrapper:
    if str(config.task_type).strip().lower() != "classification":
        raise ValueError("build_binary_ensemble_wrapper only supports classification configs")
    return build_ensemble_wrapper(config, selection_checkpoints, trajectory_checkpoints)


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
    task_type: str = "classification",
    ctb_target_modes: Sequence[str] = ("loss_aware",),
    ctb_curvature_eps: Sequence[float] = (1e-6,),
) -> List[EnsembleModelConfig]:
    grid: List[EnsembleModelConfig] = []
    for family in families:
        family_name = normalize_ctb_tree_family_name(family)
        if family_name == "ctb":
            for depth in max_depths:
                for target_mode in ctb_target_modes:
                    for curvature_eps in ctb_curvature_eps:
                        grid.append(
                            EnsembleModelConfig(
                                family=family_name,
                                max_depth=int(depth),
                                n_estimators=int(n_estimators),
                                task_type=str(task_type),
                                min_samples_leaf=int(min_samples_leaf),
                                learning_rate=float(learning_rate),
                                subsample=float(subsample),
                                colsample_bytree=float(colsample_bytree),
                                inner_bootstraps=int(inner_bootstraps),
                                eta=float(eta),
                                instability_penalty=float(instability_penalty),
                                weight_power=float(weight_power),
                                weight_eps=float(weight_eps),
                                ctb_target_mode=str(target_mode),
                                ctb_curvature_eps=float(curvature_eps),
                                random_state=int(random_state),
                            )
                        )
            continue
        for depth in max_depths:
            grid.append(
                EnsembleModelConfig(
                    family=family_name,
                    max_depth=int(depth),
                    n_estimators=int(n_estimators),
                    task_type=str(task_type),
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
