from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sim.ctb_core import ConsensusTransportBoosting
from sim.ctb_semantics import ctb_tree_method_aliases

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


TaskType = Literal["regression", "classification"]


@dataclass
class LearnerSpec:
    name: str
    task_type: TaskType
    estimator: object

    def build(self) -> object:
        return clone(self.estimator)


class SklearnLikeWrapper:
    def __init__(self, estimator: object, task_type: TaskType):
        self.estimator = estimator
        self.task_type = task_type

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnLikeWrapper":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = self.estimator.predict(X)
        return np.asarray(pred, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification models")
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1], dtype=float)
            return np.asarray(proba, dtype=float).reshape(-1)
        pred = self.predict(X)
        return 1.0 / (1.0 + np.exp(-pred))


def make_default_learner_specs(
    random_state: int = 0,
    *,
    ctb_n_estimators: int = 50,
    ctb_inner_bootstraps: int = 8,
    ctb_eta: float = 1.0,
    ctb_instability_penalty: float = 0.0,
    ctb_weight_power: float = 1.0,
    ctb_weight_eps: float = 1e-8,
    ctb_target_mode: str = "legacy",
    ctb_curvature_eps: float = 1e-6,
    ctb_min_samples_leaf: int = 5,
    ctb_weak_learner_backend: str = "xgb_tree",
    ctb_xgb_learning_rate: float = 0.05,
    ctb_xgb_subsample: float = 0.9,
    ctb_xgb_colsample_bytree: float = 0.9,
    ctb_xgb_reg_lambda: float = 1.0,
    ctb_xgb_min_child_weight: float = 1.0,
) -> Dict[str, LearnerSpec]:
    specs: Dict[str, LearnerSpec] = {}

    specs["bagging_tree_deep_regression"] = LearnerSpec(
        name="bagging_tree_deep_regression",
        task_type="regression",
        estimator=BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=None, min_samples_leaf=5, random_state=random_state),
            n_estimators=200,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    specs["bagging_tree_shallow_regression"] = LearnerSpec(
        name="bagging_tree_shallow_regression",
        task_type="regression",
        estimator=BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=random_state),
            n_estimators=200,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    specs["gbdt_stump_regression"] = LearnerSpec(
        name="gbdt_stump_regression",
        task_type="regression",
        estimator=GradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            n_estimators=300,
            max_depth=1,
            subsample=1.0,
            random_state=random_state,
        ),
    )
    specs["gbdt_depth3_regression"] = LearnerSpec(
        name="gbdt_depth3_regression",
        task_type="regression",
        estimator=GradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            n_estimators=300,
            max_depth=3,
            subsample=1.0,
            random_state=random_state,
        ),
    )

    ctb_stump_regression_spec = LearnerSpec(
        name=ctb_tree_method_aliases(depth=1, task_type="regression")[0],
        task_type="regression",
        estimator=ConsensusTransportBoosting(
            task_type="regression",
            n_estimators=ctb_n_estimators,
            n_inner_bootstraps=ctb_inner_bootstraps,
            eta=ctb_eta,
            instability_penalty=ctb_instability_penalty,
            weight_power=ctb_weight_power,
            weight_eps=ctb_weight_eps,
            update_target_mode=ctb_target_mode,
            transport_curvature_eps=ctb_curvature_eps,
            max_depth=1,
            min_samples_leaf=ctb_min_samples_leaf,
            weak_learner_backend=ctb_weak_learner_backend,
            xgb_learning_rate=ctb_xgb_learning_rate,
            xgb_subsample=ctb_xgb_subsample,
            xgb_colsample_bytree=ctb_xgb_colsample_bytree,
            xgb_reg_lambda=ctb_xgb_reg_lambda,
            xgb_min_child_weight=ctb_xgb_min_child_weight,
            random_state=random_state,
        ),
    )
    for alias in ctb_tree_method_aliases(depth=1, task_type="regression"):
        specs[alias] = ctb_stump_regression_spec
    ctb_depth3_regression_spec = LearnerSpec(
        name=ctb_tree_method_aliases(depth=3, task_type="regression")[0],
        task_type="regression",
        estimator=ConsensusTransportBoosting(
            task_type="regression",
            n_estimators=ctb_n_estimators,
            n_inner_bootstraps=ctb_inner_bootstraps,
            eta=ctb_eta,
            instability_penalty=ctb_instability_penalty,
            weight_power=ctb_weight_power,
            weight_eps=ctb_weight_eps,
            update_target_mode=ctb_target_mode,
            transport_curvature_eps=ctb_curvature_eps,
            max_depth=3,
            min_samples_leaf=ctb_min_samples_leaf,
            weak_learner_backend=ctb_weak_learner_backend,
            xgb_learning_rate=ctb_xgb_learning_rate,
            xgb_subsample=ctb_xgb_subsample,
            xgb_colsample_bytree=ctb_xgb_colsample_bytree,
            xgb_reg_lambda=ctb_xgb_reg_lambda,
            xgb_min_child_weight=ctb_xgb_min_child_weight,
            random_state=random_state,
        ),
    )
    for alias in ctb_tree_method_aliases(depth=3, task_type="regression"):
        specs[alias] = ctb_depth3_regression_spec

    specs["bagging_tree_deep_classification"] = LearnerSpec(
        name="bagging_tree_deep_classification",
        task_type="classification",
        estimator=BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=None, min_samples_leaf=5, random_state=random_state),
            n_estimators=200,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    specs["bagging_tree_shallow_classification"] = LearnerSpec(
        name="bagging_tree_shallow_classification",
        task_type="classification",
        estimator=BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=random_state),
            n_estimators=200,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    specs["gbdt_stump_classification"] = LearnerSpec(
        name="gbdt_stump_classification",
        task_type="classification",
        estimator=GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=300,
            max_depth=1,
            subsample=1.0,
            random_state=random_state,
        ),
    )
    specs["gbdt_depth3_classification"] = LearnerSpec(
        name="gbdt_depth3_classification",
        task_type="classification",
        estimator=GradientBoostingClassifier(
            learning_rate=0.05,
            n_estimators=300,
            max_depth=3,
            subsample=1.0,
            random_state=random_state,
        ),
    )

    ctb_stump_classification_spec = LearnerSpec(
        name=ctb_tree_method_aliases(depth=1, task_type="classification")[0],
        task_type="classification",
        estimator=ConsensusTransportBoosting(
            task_type="classification",
            n_estimators=ctb_n_estimators,
            n_inner_bootstraps=ctb_inner_bootstraps,
            eta=ctb_eta,
            instability_penalty=ctb_instability_penalty,
            weight_power=ctb_weight_power,
            weight_eps=ctb_weight_eps,
            update_target_mode=ctb_target_mode,
            transport_curvature_eps=ctb_curvature_eps,
            max_depth=1,
            min_samples_leaf=ctb_min_samples_leaf,
            weak_learner_backend=ctb_weak_learner_backend,
            xgb_learning_rate=ctb_xgb_learning_rate,
            xgb_subsample=ctb_xgb_subsample,
            xgb_colsample_bytree=ctb_xgb_colsample_bytree,
            xgb_reg_lambda=ctb_xgb_reg_lambda,
            xgb_min_child_weight=ctb_xgb_min_child_weight,
            random_state=random_state,
        ),
    )
    for alias in ctb_tree_method_aliases(depth=1, task_type="classification"):
        specs[alias] = ctb_stump_classification_spec
    ctb_depth3_classification_spec = LearnerSpec(
        name=ctb_tree_method_aliases(depth=3, task_type="classification")[0],
        task_type="classification",
        estimator=ConsensusTransportBoosting(
            task_type="classification",
            n_estimators=ctb_n_estimators,
            n_inner_bootstraps=ctb_inner_bootstraps,
            eta=ctb_eta,
            instability_penalty=ctb_instability_penalty,
            weight_power=ctb_weight_power,
            weight_eps=ctb_weight_eps,
            update_target_mode=ctb_target_mode,
            transport_curvature_eps=ctb_curvature_eps,
            max_depth=3,
            min_samples_leaf=ctb_min_samples_leaf,
            weak_learner_backend=ctb_weak_learner_backend,
            xgb_learning_rate=ctb_xgb_learning_rate,
            xgb_subsample=ctb_xgb_subsample,
            xgb_colsample_bytree=ctb_xgb_colsample_bytree,
            xgb_reg_lambda=ctb_xgb_reg_lambda,
            xgb_min_child_weight=ctb_xgb_min_child_weight,
            random_state=random_state,
        ),
    )
    for alias in ctb_tree_method_aliases(depth=3, task_type="classification"):
        specs[alias] = ctb_depth3_classification_spec

    if XGBRegressor is not None and XGBClassifier is not None:
        specs["xgboost_regression"] = LearnerSpec(
            name="xgboost_regression",
            task_type="regression",
            estimator=XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            ),
        )
        specs["xgboost_classification"] = LearnerSpec(
            name="xgboost_classification",
            task_type="classification",
            estimator=XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            ),
        )
    return specs


def build_model(
    method_name: str,
    task_type: TaskType,
    random_state: int = 0,
    **spec_kwargs,
) -> SklearnLikeWrapper:
    specs = make_default_learner_specs(random_state=random_state, **spec_kwargs)
    if method_name not in specs:
        available = sorted(specs)
        raise KeyError(f"Unknown method_name={method_name!r}. Available: {available}")
    spec = specs[method_name]
    if spec.task_type != task_type:
        raise ValueError(f"Method {method_name!r} is for task_type={spec.task_type!r}, got {task_type!r}")
    return SklearnLikeWrapper(estimator=spec.build(), task_type=task_type)


def default_methods_for_task(task_type: TaskType) -> list[str]:
    if task_type == "regression":
        return [
            "bagging_tree_deep_regression",
            "bagging_tree_shallow_regression",
            "gbdt_stump_regression",
            "gbdt_depth3_regression",
            "ctb_stump_regression",
            "ctb_depth3_regression",
        ]
    return [
        "bagging_tree_deep_classification",
        "bagging_tree_shallow_classification",
        "gbdt_stump_classification",
        "gbdt_depth3_classification",
        "ctb_stump_classification",
        "ctb_depth3_classification",
    ]
