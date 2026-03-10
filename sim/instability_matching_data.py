from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

Array = np.ndarray
TaskType = Literal["regression", "classification"]
ScenarioType = Literal["piecewise", "smooth", "pocket"]
FeatureDistType = Literal["uniform", "gaussian"]
NoiseType = Literal["homoscedastic", "heteroscedastic"]


@dataclass(frozen=True)
class SplitData:
    X: Array
    y: Array
    f_true: Array
    meta: Dict[str, Array]


@dataclass(frozen=True)
class DatasetBundle:
    task_type: TaskType
    scenario: ScenarioType
    feature_dist: FeatureDistType
    noise_type: NoiseType
    train: SplitData
    valid: SplitData
    test: SplitData
    config: Dict[str, object]


DEFAULT_THRESHOLDS: Tuple[Tuple[int, float], ...] = ((0, 0.2), (1, -0.3), (2, 0.5))


def sigmoid(z: Array) -> Array:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))



def generate_features(
    n_samples: int,
    p: int = 20,
    feature_dist: FeatureDistType = "uniform",
    rng: Optional[np.random.Generator] = None,
) -> Array:
    if rng is None:
        rng = np.random.default_rng()

    if feature_dist == "uniform":
        return rng.uniform(-1.0, 1.0, size=(n_samples, p))
    if feature_dist == "gaussian":
        return rng.normal(0.0, 1.0, size=(n_samples, p))
    raise ValueError(f"Unsupported feature_dist={feature_dist!r}")



def signal_piecewise_constant(X: Array) -> Array:
    return (
        2.0 * (X[:, 0] > 0.2).astype(float)
        + 1.5 * (X[:, 1] < -0.3).astype(float)
        - 2.0 * (X[:, 2] > 0.5).astype(float)
    )



def signal_smooth_additive(X: Array) -> Array:
    return 2.0 * np.sin(np.pi * X[:, 0]) + 1.5 * (X[:, 1] ** 2) + X[:, 2] - 1.2 * np.abs(X[:, 3])



def signal_rare_local_pocket(X: Array) -> Array:
    pocket = np.exp(-((X[:, 1] - 0.5) ** 2 + (X[:, 2] + 0.4) ** 2) / 0.02)
    return 1.5 * X[:, 0] + 4.0 * pocket



def generate_latent_signal(X: Array, scenario: ScenarioType) -> Array:
    if scenario == "piecewise":
        return signal_piecewise_constant(X)
    if scenario == "smooth":
        return signal_smooth_additive(X)
    if scenario == "pocket":
        return signal_rare_local_pocket(X)
    raise ValueError(f"Unsupported scenario={scenario!r}")



def conditional_noise_std(
    X: Array,
    noise_type: NoiseType = "homoscedastic",
    base_scale: float = 0.5,
) -> Array:
    if noise_type == "homoscedastic":
        return np.full(X.shape[0], base_scale, dtype=float)
    if noise_type == "heteroscedastic":
        return base_scale * (0.4 + 1.6 * np.abs(X[:, 0]))
    raise ValueError(f"Unsupported noise_type={noise_type!r}")



def generate_regression_targets(f_true: Array, sigma_true: Array, rng: np.random.Generator) -> Array:
    eps = rng.normal(loc=0.0, scale=sigma_true, size=f_true.shape[0])
    return f_true + eps



def generate_classification_targets(f_true: Array, rng: np.random.Generator) -> Tuple[Array, Array]:
    prob_true = sigmoid(f_true)
    y = rng.binomial(1, prob_true, size=f_true.shape[0]).astype(int)
    return y, prob_true



def make_oracle_metadata(
    X: Array,
    scenario: ScenarioType,
    task_type: TaskType,
    f_true: Array,
    sigma_true: Array,
) -> Dict[str, Array]:
    meta: Dict[str, Array] = {
        "f_true": f_true.astype(float),
        "sigma_true": sigma_true.astype(float),
        "high_noise": (sigma_true >= np.quantile(sigma_true, 0.75)).astype(int),
    }

    if task_type == "classification":
        prob_true = sigmoid(f_true)
        margin_true = np.abs(prob_true - 0.5)
        meta["prob_true"] = prob_true.astype(float)
        meta["margin_true"] = margin_true.astype(float)
        meta["small_margin"] = (margin_true <= np.quantile(margin_true, 0.25)).astype(int)

    if scenario == "piecewise":
        d0 = np.abs(X[:, 0] - 0.2)
        d1 = np.abs(X[:, 1] + 0.3)
        d2 = np.abs(X[:, 2] - 0.5)
        threshold_dist = np.minimum(np.minimum(d0, d1), d2)
        meta["threshold_dist"] = threshold_dist.astype(float)
        meta["near_threshold"] = (threshold_dist < 0.1).astype(int)

    if scenario == "smooth":
        curvature_proxy = np.abs(-2.0 * (np.pi**2) * np.sin(np.pi * X[:, 0])) + 3.0
        meta["curvature_proxy"] = curvature_proxy.astype(float)
        meta["high_curvature"] = (curvature_proxy >= np.quantile(curvature_proxy, 0.75)).astype(int)

    if scenario == "pocket":
        pocket_sq_dist = (X[:, 1] - 0.5) ** 2 + (X[:, 2] + 0.4) ** 2
        pocket_strength = np.exp(-pocket_sq_dist / 0.02)
        meta["pocket_sq_dist"] = pocket_sq_dist.astype(float)
        meta["pocket_strength"] = pocket_strength.astype(float)
        meta["is_pocket"] = (pocket_sq_dist < 0.05).astype(int)

    return meta



def _build_split(
    n_samples: int,
    task_type: TaskType,
    scenario: ScenarioType,
    feature_dist: FeatureDistType,
    noise_type: NoiseType,
    noise_scale: float,
    p: int,
    rng: np.random.Generator,
) -> SplitData:
    X = generate_features(n_samples=n_samples, p=p, feature_dist=feature_dist, rng=rng)
    f_true = generate_latent_signal(X, scenario=scenario)
    sigma_true = conditional_noise_std(X, noise_type=noise_type, base_scale=noise_scale)

    if task_type == "regression":
        y = generate_regression_targets(f_true, sigma_true, rng=rng)
    elif task_type == "classification":
        y, _ = generate_classification_targets(f_true, rng=rng)
    else:
        raise ValueError(f"Unsupported task_type={task_type!r}")

    meta = make_oracle_metadata(X=X, scenario=scenario, task_type=task_type, f_true=f_true, sigma_true=sigma_true)
    return SplitData(X=X.astype(float), y=y, f_true=f_true.astype(float), meta=meta)



def generate_dataset_bundle(
    task_type: TaskType,
    scenario: ScenarioType,
    n_train: int = 500,
    n_valid: int = 500,
    n_test: int = 5000,
    p: int = 20,
    feature_dist: FeatureDistType = "uniform",
    noise_type: NoiseType = "homoscedastic",
    noise_scale: float = 0.5,
    seed: int = 0,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    train = _build_split(
        n_samples=n_train,
        task_type=task_type,
        scenario=scenario,
        feature_dist=feature_dist,
        noise_type=noise_type,
        noise_scale=noise_scale,
        p=p,
        rng=rng,
    )
    valid = _build_split(
        n_samples=n_valid,
        task_type=task_type,
        scenario=scenario,
        feature_dist=feature_dist,
        noise_type=noise_type,
        noise_scale=noise_scale,
        p=p,
        rng=rng,
    )
    test = _build_split(
        n_samples=n_test,
        task_type=task_type,
        scenario=scenario,
        feature_dist=feature_dist,
        noise_type=noise_type,
        noise_scale=noise_scale,
        p=p,
        rng=rng,
    )

    config = {
        "task_type": task_type,
        "scenario": scenario,
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "p": p,
        "feature_dist": feature_dist,
        "noise_type": noise_type,
        "noise_scale": noise_scale,
        "seed": seed,
    }
    return DatasetBundle(
        task_type=task_type,
        scenario=scenario,
        feature_dist=feature_dist,
        noise_type=noise_type,
        train=train,
        valid=valid,
        test=test,
        config=config,
    )



def summarize_dataset_bundle(bundle: DatasetBundle) -> Dict[str, float]:
    train_meta = bundle.train.meta
    test_meta = bundle.test.meta
    summary: Dict[str, float] = {
        "train_n": float(bundle.train.X.shape[0]),
        "valid_n": float(bundle.valid.X.shape[0]),
        "test_n": float(bundle.test.X.shape[0]),
        "train_f_mean": float(bundle.train.f_true.mean()),
        "train_f_std": float(bundle.train.f_true.std()),
        "train_sigma_mean": float(train_meta["sigma_true"].mean()),
        "test_sigma_mean": float(test_meta["sigma_true"].mean()),
    }

    if bundle.task_type == "regression":
        summary["train_y_mean"] = float(bundle.train.y.mean())
        summary["train_y_std"] = float(bundle.train.y.std())
    else:
        summary["train_positive_rate"] = float(bundle.train.y.mean())
        summary["test_positive_rate"] = float(bundle.test.y.mean())
        summary["test_small_margin_rate"] = float(test_meta["small_margin"].mean())

    if bundle.scenario == "piecewise":
        summary["test_near_threshold_rate"] = float(test_meta["near_threshold"].mean())
    elif bundle.scenario == "smooth":
        summary["test_high_curvature_rate"] = float(test_meta["high_curvature"].mean())
    elif bundle.scenario == "pocket":
        summary["test_pocket_rate"] = float(test_meta["is_pocket"].mean())
        summary["test_pocket_strength_mean"] = float(test_meta["pocket_strength"].mean())

    summary["test_high_noise_rate"] = float(test_meta["high_noise"].mean())
    return summary
