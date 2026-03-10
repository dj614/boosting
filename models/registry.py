from __future__ import annotations

from typing import Dict, Type

from models.baselines import GroupedPartialLinearBaseline, RandomForestConformalRegressor
from models.base import InferenceModel, PredictionModel

PREDICTION_MODELS: Dict[str, Type[PredictionModel]] = {
    "rf_conformal": RandomForestConformalRegressor,
    "random_forest_conformal": RandomForestConformalRegressor,
    "RandomForestConformalRegressor": RandomForestConformalRegressor,
}

INFERENCE_MODELS: Dict[str, Type[InferenceModel]] = {
    "grouped_partial_linear_baseline": GroupedPartialLinearBaseline,
    "GroupedPartialLinearBaseline": GroupedPartialLinearBaseline,
}