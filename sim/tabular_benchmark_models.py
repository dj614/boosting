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
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .ctb_core import ConsensusTransportBoosting

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
    random_state: int = 0

    @property
    def model_name(self) -> str:
        return f"{self.family}_depth{self.max_depth}"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class TabularBenchmarkWrapper:
    def __init__(self, config: TabularBenchmarkModelConfig, selection_checkpoints: Sequence[int]) -> None:
        self.config = config
        self.selection_checkpoints = sorted({int(x) for x in selection_checkpoints if int(x) > 0})
        if not self.selection_checkpoints:
            raise ValueError("selection_checkpoints must be non-empty")
        self.model = None
        self.selected_checkpoint_: Optional[int] = None
        self.selection_trace_: Optional[pd.DataFrame] = None

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def fit(self, train_split, valid_split) -> "TabularBenchmarkWrapper":
        self.model = self._build_estimator()
        self.model.fit(train_split.X, train_split.y)
        staged_valid_predictions = self._predict_at_checkpoints(valid_split.X, checkpoints=self.selection_checkpoints)

        rows: List[Dict[str, float]] = []
        best_checkpoint = None
        best_score = None
        for checkpoint in self.selection_checkpoints:
            pred = np.asarray(staged_valid_predictions[int(checkpoint)], dtype=float)
            current_score = float(self._selection_score(valid_split.y, pred))
            rows.append({"checkpoint": int(checkpoint), self.selection_metric_name: current_score})
            if best_score is None or current_score < best_score - 1e-12 or (
                abs(current_score - best_score) <= 1e-12 and int(checkpoint) < int(best_checkpoint)
            ):
                best_score = current_score
                best_checkpoint = int(checkpoint)
        self.selected_checkpoint_ = int(best_checkpoint)
        self.selection_trace_ = pd.DataFrame(rows)
        return self

    @property
    def selection_metric_name(self) -> str:
        raise NotImplementedError

    def _selection_score(self, y_true: Array, prediction: Array) -> float:
        raise NotImplementedError

    def _build_estimator(self):
        raise NotImplementedError

    def _predict_at_checkpoints(self, X: Array, checkpoints: Sequence[int]) -> Dict[int, Array]:
        raise NotImplementedError


class BinaryTabularBenchmarkWrapper(TabularBenchmarkWrapper):
    @property
    def selection_metric_name(self) -> str:
        return "valid_log_loss"

    def _selection_score(self, y_true: Array, prediction: Array) -> float:
        return float(log_loss(np.asarray(y_true, dtype=int), np.clip(prediction, 1e-8, 1.0 - 1e-8), labels=[0, 1]))

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
        return "valid_rmse"

    def _selection_score(self, y_true: Array, prediction: Array) -> float:
        y_true = np.asarray(y_true, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        return float(np.sqrt(np.mean((y_true - prediction) ** 2)))

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
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
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
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
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
) -> TabularBenchmarkWrapper:
    task_type = str(config.task_type).strip().lower()
    family = str(config.family).strip().lower()
    if task_type == "classification":
        if family == "bagging":
            return BaggingBinaryTabularWrapper(config, selection_checkpoints)
        if family == "rf":
            return RandomForestBinaryTabularWrapper(config, selection_checkpoints)
        if family == "gbdt":
            return GradientBoostingBinaryTabularWrapper(config, selection_checkpoints)
        if family == "xgb":
            return XGBoostBinaryTabularWrapper(config, selection_checkpoints)
        if family == "ctb":
            return CTBBinaryTabularWrapper(config, selection_checkpoints)
    elif task_type == "regression":
        if family == "bagging":
            return BaggingRegressionTabularWrapper(config, selection_checkpoints)
        if family == "rf":
            return RandomForestRegressionTabularWrapper(config, selection_checkpoints)
        if family == "gbdt":
            return GradientBoostingRegressionTabularWrapper(config, selection_checkpoints)
        if family == "xgb":
            return XGBoostRegressionTabularWrapper(config, selection_checkpoints)
        if family == "ctb":
            return CTBRegressionTabularWrapper(config, selection_checkpoints)
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
        "learning_rate": (0.1,),
        "subsample": (1.0,),
        "colsample_bytree": (0.8,),
    },
}


def expand_tabular_model_grid(
    task_type: str,
    families: Sequence[str],
    max_depths: Sequence[int],
    n_estimators: int,
    min_samples_leafs: Sequence[int] = (1, 5),
    learning_rates: Sequence[float] = (0.03, 0.1),
    subsamples: Sequence[float] = (0.7, 1.0),
    colsample_bytree: Sequence[float] = (0.8,),
    inner_bootstraps: Sequence[int] = (4, 8),
    etas: Sequence[float] = (0.5, 1.0),
    instability_penalty: float = 0.0,
    weight_power: float = 1.0,
    weight_eps: float = 1e-8,
    random_state: int = 0,
) -> List[TabularBenchmarkModelConfig]:
    grid: List[TabularBenchmarkModelConfig] = []
    task_type = str(task_type).strip().lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task_type={task_type!r}")

    for family_name in families:
        family = str(family_name).strip().lower()
        if family not in FAMILY_DEFAULTS:
            raise ValueError(f"Unsupported family={family!r}")
        default_overrides = FAMILY_DEFAULTS[family]
        family_learning_rates = tuple(default_overrides.get("learning_rate", learning_rates))
        family_subsamples = tuple(default_overrides.get("subsample", subsamples))
        family_colsample = tuple(default_overrides.get("colsample_bytree", colsample_bytree))
        family_inner_bootstraps = tuple(default_overrides.get("inner_bootstraps", inner_bootstraps))
        family_etas = tuple(default_overrides.get("eta", etas))

        if family in {"bagging", "rf"}:
            for depth in max_depths:
                for leaf in min_samples_leafs:
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
                            random_state=int(random_state),
                        )
                    )
            continue

        if family in {"gbdt", "xgb"}:
            for depth in max_depths:
                for lr in family_learning_rates:
                    for subsample in family_subsamples:
                        for colsample in family_colsample:
                            grid.append(
                                TabularBenchmarkModelConfig(
                                    task_type=task_type,
                                    family=family,
                                    max_depth=int(depth),
                                    n_estimators=int(n_estimators),
                                    min_samples_leaf=int(min_samples_leafs[0]),
                                    learning_rate=float(lr),
                                    subsample=float(subsample),
                                    colsample_bytree=float(colsample),
                                    inner_bootstraps=int(family_inner_bootstraps[0]),
                                    eta=float(family_etas[0]),
                                    instability_penalty=float(instability_penalty),
                                    weight_power=float(weight_power),
                                    weight_eps=float(weight_eps),
                                    random_state=int(random_state),
                                )
                            )
            continue
        if family == "ctb":
            for depth in max_depths:
                for leaf in min_samples_leafs:
                    for inner_bootstrap in family_inner_bootstraps:
                        for eta in family_etas:
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