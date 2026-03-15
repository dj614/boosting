from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
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
class TabularBenchmarkModelConfig:
    task_type: str
    family: str
    max_depth: int
    n_estimators: int
    min_samples_leaf: int = 5
    learning_rate: float = 0.1
    subsample: float = 1.0
    colsample_bytree: float = 0.8
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
                include_task_suffix=False,
            )
        return f"{family}_depth{self.max_depth}"

    @property
    def family_output_name(self) -> str:
        return ctb_family_output_name(
            family_name=self.family,
            weak_learner_backend=self.ctb_weak_learner_backend,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class TabularBenchmarkWrapper:
    def __init__(
        self,
        config: TabularBenchmarkModelConfig,
        selection_checkpoints: Sequence[int],
        use_report_metric_for_selection: bool = False,
    ) -> None:
        self.config = config
        self.selection_checkpoints = sorted({int(x) for x in selection_checkpoints if int(x) > 0})
        if not self.selection_checkpoints:
            raise ValueError("selection_checkpoints must be non-empty")
        self.use_report_metric_for_selection = bool(use_report_metric_for_selection)
        self.model = None
        self.selected_checkpoint_: Optional[int] = None
        self.selection_trace_: Optional[pd.DataFrame] = None
        self.selected_valid_prediction_: Optional[Array] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(self, train_split, valid_split) -> "TabularBenchmarkWrapper":
        self.model = self._build_estimator()
        self.model.fit(train_split.X, train_split.y)
        staged_valid_predictions = self._predict_at_checkpoints(valid_split.X, checkpoints=self.selection_checkpoints)

        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_metric = None
        for checkpoint in self.selection_checkpoints:
            pred = np.asarray(staged_valid_predictions[int(checkpoint)], dtype=float)
            current_metric = float(self._selection_metric_value(valid_split.y, pred))
            rows.append({"checkpoint": int(checkpoint), self.selection_metric_name: current_metric})
            if self._is_better_metric(current_metric=current_metric, best_metric=best_metric, best_checkpoint=best_checkpoint, checkpoint=checkpoint):
                best_metric = current_metric
                best_checkpoint = int(checkpoint)
        self.selected_checkpoint_ = int(best_checkpoint)
        self.selected_valid_prediction_ = np.asarray(
            staged_valid_predictions[int(self.selected_checkpoint_)],
            dtype=float,
        ).copy()
        self.selection_trace_ = pd.DataFrame(rows)
        return self

    @property
    def selection_metric_name(self) -> str:
        raise NotImplementedError

    @property
    def selection_metric_higher_is_better(self) -> bool:
        return False

    def _selection_metric_value(self, y_true: Array, prediction: Array) -> float:
        raise NotImplementedError

    def _is_better_metric(
        self,
        *,
        current_metric: float,
        best_metric: Optional[float],
        best_checkpoint: Optional[int],
        checkpoint: int,
    ) -> bool:
        if best_metric is None:
            return True
        if self.selection_metric_higher_is_better:
            return current_metric > best_metric + 1e-12 or (
                abs(current_metric - best_metric) <= 1e-12 and int(checkpoint) < int(best_checkpoint)
            )
        return current_metric < best_metric - 1e-12 or (
            abs(current_metric - best_metric) <= 1e-12 and int(checkpoint) < int(best_checkpoint)
        )

    def _build_estimator(self):
        raise NotImplementedError

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        raise NotImplementedError


class BinaryTabularBenchmarkWrapper(TabularBenchmarkWrapper):
    @property
    def selection_metric_name(self) -> str:
        if self.use_report_metric_for_selection:
            return "valid_accuracy"
        return "valid_log_loss"

    @property
    def selection_metric_higher_is_better(self) -> bool:
        return bool(self.use_report_metric_for_selection)

    def _selection_metric_value(self, y_true: Array, prediction: Array) -> float:
        y_true_arr = np.asarray(y_true, dtype=int)
        prob = np.clip(np.asarray(prediction, dtype=float), 1e-8, 1.0 - 1e-8)
        if self.use_report_metric_for_selection:
            return float(accuracy_score(y_true_arr, (prob >= 0.5).astype(int)))
        return float(log_loss(y_true_arr, prob, labels=[0, 1]))

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


class RegressionTabularBenchmarkWrapper(TabularBenchmarkWrapper):
    @property
    def selection_metric_name(self) -> str:
        if self.use_report_metric_for_selection:
            return "valid_mse"
        return "valid_rmse"

    def _selection_metric_value(self, y_true: Array, prediction: Array) -> float:
        y_true = np.asarray(y_true, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        mse = float(mean_squared_error(y_true, prediction))
        if self.use_report_metric_for_selection:
            return mse
        return float(np.sqrt(mse))

    def predict(self, X: Array, checkpoint: Optional[int] = None) -> Array:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        use_checkpoint = int(checkpoint or self.selected_checkpoint_ or self.config.n_estimators)
        return np.asarray(self._predict_at_checkpoints(X, checkpoints=[use_checkpoint])[use_checkpoint], dtype=float)

    def predict_staged(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        if self.model is None:
            raise RuntimeError("Model has not been fit")
        return self._predict_at_checkpoints(X, checkpoints=checkpoints)


class BaggingBinaryTabularWrapper(BinaryTabularBenchmarkWrapper):
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


class RandomForestBinaryTabularWrapper(BinaryTabularBenchmarkWrapper):
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


class GradientBoostingBinaryTabularWrapper(BinaryTabularBenchmarkWrapper):
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


class XGBoostBinaryTabularWrapper(BinaryTabularBenchmarkWrapper):
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


class CTBBinaryTabularWrapper(BinaryTabularBenchmarkWrapper):
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


class BaggingRegressionTabularWrapper(RegressionTabularBenchmarkWrapper):
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
        estimator_preds = [np.asarray(est.predict(X), dtype=float).reshape(-1) for est in self.model.estimators_]
        return _prefix_average_predictions(estimator_outputs=estimator_preds, checkpoints=checkpoints)


class RandomForestRegressionTabularWrapper(RegressionTabularBenchmarkWrapper):
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
        estimator_preds = [np.asarray(est.predict(X), dtype=float).reshape(-1) for est in self.model.estimators_]
        return _prefix_average_predictions(estimator_outputs=estimator_preds, checkpoints=checkpoints)


class GradientBoostingRegressionTabularWrapper(RegressionTabularBenchmarkWrapper):
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
                out[idx] = np.asarray(pred, dtype=float).reshape(-1)
            if len(out) == len(requested):
                break
        _validate_checkpoint_outputs(requested, out, self.config.n_estimators)
        return out


class XGBoostRegressionTabularWrapper(RegressionTabularBenchmarkWrapper):
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
            min_child_weight=1.0,
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
            out[int(checkpoint)] = np.asarray(pred, dtype=float).reshape(-1)
        return out


class CTBRegressionTabularWrapper(RegressionTabularBenchmarkWrapper):
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
            out[int(checkpoint)] = np.asarray(pred, dtype=float).reshape(-1)
        _validate_checkpoint_outputs(requested, out, self.config.n_estimators)
        return out


def build_tabular_benchmark_wrapper(
    config: TabularBenchmarkModelConfig,
    selection_checkpoints: Sequence[int],
    use_report_metric_for_selection: bool = False,
) -> TabularBenchmarkWrapper:
    task_type = str(config.task_type).strip().lower()
    family = normalize_ctb_tree_family_name(config.family)
    if task_type == "classification":
        if family == "bagging":
            return BaggingBinaryTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
        if family == "rf":
            return RandomForestBinaryTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
        if family == "gbdt":
            return GradientBoostingBinaryTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
        if family == "xgb":
            return XGBoostBinaryTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
        if family == "ctb":
            return CTBBinaryTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
    elif task_type == "regression":
        if family == "bagging":
            return BaggingRegressionTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
        if family == "rf":
            return RandomForestRegressionTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
        if family == "gbdt":
            return GradientBoostingRegressionTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
        if family == "xgb":
            return XGBoostRegressionTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
        if family == "ctb":
            return CTBRegressionTabularWrapper(config, selection_checkpoints, use_report_metric_for_selection)
    raise ValueError(f"Unsupported task_type={task_type!r} family={family!r}")


FAMILY_DEFAULTS = {
    "bagging": {
        "learning_rate": (0.1,),
        "subsample": (1.0,),
        "colsample_bytree": (0.8,),
        "inner_bootstraps": (8,),
        "eta": (1.0,),
    },
    "rf": {
        "learning_rate": (0.1,),
        "subsample": (1.0,),
        "colsample_bytree": (0.8,),
        "inner_bootstraps": (8,),
        "eta": (1.0,),
    },
    "gbdt": {
        "inner_bootstraps": (8,),
        "eta": (1.0,),
    },
    "xgb": {
        "inner_bootstraps": (8,),
        "eta": (1.0,),
    },
    "ctb": {
        "max_depths": (1, 3),
        "min_samples_leafs": (5,),
        "learning_rate": (0.1,),
        "subsample": (1.0,),
        "colsample_bytree": (0.8,),
        "inner_bootstraps": (2,),
        "eta": (1.0,),
        "ctb_target_mode": ("loss_aware",),
        "ctb_weak_learner_backend": ("xgb_tree",),
    },
}


def expand_tabular_model_grid(
    task_type: str,
    families: Sequence[str],
    n_estimators: int,
    max_depths: Sequence[int] | None = None,
    min_samples_leafs: Sequence[int] | None = None,
    learning_rates: Sequence[float] | None = None,
    subsamples: Sequence[float] | None = None,
    colsample_bytree: Sequence[float] | None = None,
    inner_bootstraps: Sequence[int] | None = None,
    etas: Sequence[float] | None = None,
    instability_penalty: float = 0.0,
    weight_power: float = 1.0,
    weight_eps: float = 1e-8,
    ctb_target_modes: Sequence[str] | None = None,
    ctb_weak_learner_backends: Sequence[str] | None = None,
    ctb_curvature_eps: Sequence[float] | None = None,
    random_state: int = 0,
) -> List[TabularBenchmarkModelConfig]:
    grid: List[TabularBenchmarkModelConfig] = []
    task_type = str(task_type).strip().lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task_type={task_type!r}")

    default_max_depths = (1, 3, 5)
    default_min_samples_leafs = (1, 5)
    default_learning_rates = (0.03, 0.1)
    default_subsamples = (0.7, 1.0)
    default_colsample_bytree = (0.8,)
    default_inner_bootstraps = (4, 8)
    default_etas = (0.5, 1.0)
    default_ctb_target_modes = ("loss_aware",)
    default_ctb_weak_learner_backends = ("sklearn_tree",)
    default_ctb_curvature_eps = (1e-6,)

    def _resolve_grid_values(
        explicit_values: Sequence[object] | None,
        family_defaults: Mapping[str, Sequence[object]],
        key: str,
        fallback_values: Sequence[object],
    ) -> Tuple[object, ...]:
        if explicit_values is not None:
            return tuple(explicit_values)
        return tuple(family_defaults.get(key, fallback_values))

    for family_name in families:
        family = normalize_ctb_tree_family_name(family_name)
        if family not in FAMILY_DEFAULTS:
            raise ValueError(f"Unsupported family={family!r}")
        default_overrides = FAMILY_DEFAULTS[family]
        family_max_depths = _resolve_grid_values(max_depths, default_overrides, "max_depths", default_max_depths)
        family_min_samples_leafs = _resolve_grid_values(min_samples_leafs, default_overrides, "min_samples_leafs", default_min_samples_leafs)
        family_learning_rates = _resolve_grid_values(learning_rates, default_overrides, "learning_rate", default_learning_rates)
        family_subsamples = _resolve_grid_values(subsamples, default_overrides, "subsample", default_subsamples)
        family_colsample = _resolve_grid_values(colsample_bytree, default_overrides, "colsample_bytree", default_colsample_bytree)
        family_inner_bootstraps = _resolve_grid_values(inner_bootstraps, default_overrides, "inner_bootstraps", default_inner_bootstraps)
        family_etas = _resolve_grid_values(etas, default_overrides, "eta", default_etas)
        family_ctb_target_modes = _resolve_grid_values(ctb_target_modes, default_overrides, "ctb_target_mode", default_ctb_target_modes)
        family_ctb_weak_learner_backends = _resolve_grid_values(
            ctb_weak_learner_backends,
            default_overrides,
            "ctb_weak_learner_backend",
            default_ctb_weak_learner_backends,
        )
        family_ctb_curvature_eps = _resolve_grid_values(ctb_curvature_eps, default_overrides, "ctb_curvature_eps", default_ctb_curvature_eps)
        if family != "ctb":
            family_ctb_target_modes = (str(family_ctb_target_modes[0]),)
            family_ctb_weak_learner_backends = (str(family_ctb_weak_learner_backends[0]),)
            family_ctb_curvature_eps = (float(family_ctb_curvature_eps[0]),)

        if family in {"bagging", "rf"}:
            for depth in family_max_depths:
                for leaf in family_min_samples_leafs:
                    grid.append(
                        TabularBenchmarkModelConfig(
                            task_type=task_type,
                            family=family,
                            max_depth=int(depth),
                            n_estimators=int(n_estimators),
                            min_samples_leaf=int(leaf),
                            learning_rate=float(family_learning_rates[0]),
                            subsample=float(family_subsamples[0]),
                            colsample_bytree=float(family_colsample[0]),
                            inner_bootstraps=int(family_inner_bootstraps[0]),
                            eta=float(family_etas[0]),
                            instability_penalty=float(instability_penalty),
                            weight_power=float(weight_power),
                            weight_eps=float(weight_eps),
                            ctb_target_mode=str(family_ctb_target_modes[0]),
                            ctb_curvature_eps=float(family_ctb_curvature_eps[0]),
                            ctb_weak_learner_backend=str(family_ctb_weak_learner_backends[0]),
                            random_state=int(random_state),
                        )
                    )
            continue

        if family in {"gbdt", "xgb"}:
            for depth in family_max_depths:
                for lr in family_learning_rates:
                    for subsample in family_subsamples:
                        for colsample in family_colsample:
                            grid.append(
                                TabularBenchmarkModelConfig(
                                    task_type=task_type,
                                    family=family,
                                    max_depth=int(depth),
                                    n_estimators=int(n_estimators),
                                    min_samples_leaf=int(family_min_samples_leafs[0]),
                                    learning_rate=float(lr),
                                    subsample=float(subsample),
                                    colsample_bytree=float(colsample),
                                    inner_bootstraps=int(family_inner_bootstraps[0]),
                                    eta=float(family_etas[0]),
                                    instability_penalty=float(instability_penalty),
                                    weight_power=float(weight_power),
                                    weight_eps=float(weight_eps),
                                    ctb_weak_learner_backend=str(family_ctb_weak_learner_backends[0]),
                                    random_state=int(random_state),
                                )
                            )
            continue
        if family == "ctb":
            for depth in family_max_depths:
                for leaf in family_min_samples_leafs:
                    for inner_bootstrap in family_inner_bootstraps:
                        for eta in family_etas:
                            for target_mode in family_ctb_target_modes:
                                for weak_learner_backend in family_ctb_weak_learner_backends:
                                    for curvature_eps in family_ctb_curvature_eps:
                                        grid.append(
                                            TabularBenchmarkModelConfig(
                                            task_type=task_type,
                                            family=family,
                                            max_depth=int(depth),
                                            n_estimators=int(n_estimators),
                                            min_samples_leaf=int(leaf),
                                            learning_rate=float(family_learning_rates[0]),
                                            subsample=float(family_subsamples[0]),
                                            colsample_bytree=float(family_colsample[0]),
                                            inner_bootstraps=int(inner_bootstrap),
                                            eta=float(eta),
                                            instability_penalty=float(instability_penalty),
                                            weight_power=float(weight_power),
                                            weight_eps=float(weight_eps),
                                            ctb_target_mode=str(target_mode),
                                            ctb_curvature_eps=float(curvature_eps),
                                            ctb_weak_learner_backend=str(weak_learner_backend),
                                            random_state=int(random_state),
                                        )
                                    )
            continue

    return grid

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
            f"Requested checkpoint {max_requested}, but estimator only has {len(estimator_outputs)} fitted learners"
        )

    cumulative = np.zeros_like(np.asarray(estimator_outputs[0], dtype=float), dtype=float)
    out: Dict[int, Array] = {}
    requested_set = set(requested)
    for idx, pred in enumerate(estimator_outputs, start=1):
        cumulative = cumulative + np.asarray(pred, dtype=float)
        if idx in requested_set:
            out[idx] = cumulative / float(idx)
        if len(out) == len(requested):
            break
    _validate_checkpoint_outputs(requested, out, len(estimator_outputs))
    return out