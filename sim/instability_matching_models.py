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


def make_default_learner_specs(random_state: int = 0) -> Dict[str, LearnerSpec]:
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


def build_model(method_name: str, task_type: TaskType, random_state: int = 0) -> SklearnLikeWrapper:
    specs = make_default_learner_specs(random_state=random_state)
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
        ]
    return [
        "bagging_tree_deep_classification",
        "bagging_tree_shallow_classification",
        "gbdt_stump_classification",
        "gbdt_depth3_classification",
    ]
