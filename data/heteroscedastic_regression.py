from __future__ import annotations

from typing import Dict

import numpy as np

from data.types import HeteroscedasticRegressionDataset, HeteroscedasticRegressionSplit

Array = np.ndarray


def _sigmoid(z: Array) -> Array:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _nonlinear_signal(X: Array) -> Array:
    n_features = X.shape[1]
    x0 = X[:, 0]
    x1 = X[:, 1] if n_features > 1 else 0.0
    x2 = X[:, 2] if n_features > 2 else 0.0
    x3 = X[:, 3] if n_features > 3 else 0.0
    x4 = X[:, 4] if n_features > 4 else 0.0
    x5 = X[:, 5] if n_features > 5 else 0.0
    return (
        1.5 * np.sin(np.pi * x0)
        + 1.0 * x1 * x2
        + 0.8 * (x3 ** 2)
        - 1.2 * (x4 > 0.0).astype(float)
        + 0.6 * np.cos(np.pi * x5)
    )


def _heteroscedastic_noise_std(
    X: Array,
    base_noise: float,
    hetero_strength: float,
) -> Array:
    score = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.75 * X[:, 2] * X[:, 3]
    local_scale = 0.5 + hetero_strength * _sigmoid(score)
    return np.maximum(base_noise * local_scale, 1e-6)


def _build_split(
    n_samples: int,
    n_features: int,
    base_noise: float,
    hetero_strength: float,
    rng: np.random.Generator,
    split_name: str,
) -> HeteroscedasticRegressionSplit:
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features)).astype(float)
    f_true = _nonlinear_signal(X).astype(float)
    sigma_true = _heteroscedastic_noise_std(X, base_noise=base_noise, hetero_strength=hetero_strength).astype(float)
    y = (f_true + rng.normal(loc=0.0, scale=sigma_true, size=n_samples)).astype(float)
    sigma_cut = float(np.quantile(sigma_true, 0.67))
    metadata: Dict[str, Array] = {
        "noise_score": (sigma_true / max(base_noise, 1e-8)).astype(float),
        "high_noise": (sigma_true >= sigma_cut).astype(int),
    }
    sample_id = np.asarray([f"{split_name}_{i}" for i in range(n_samples)], dtype=object)
    return HeteroscedasticRegressionSplit(
        X=X,
        y=y,
        f_true=f_true,
        sigma_true=sigma_true,
        sample_id=sample_id,
        metadata=metadata,
    )


def generate_heteroscedastic_regression_dataset(
    n_train: int = 600,
    n_valid: int = 200,
    n_calib: int = 200,
    n_test: int = 1000,
    n_features: int = 8,
    base_noise: float = 0.75,
    hetero_strength: float = 1.25,
    seed: int = 0,
) -> HeteroscedasticRegressionDataset:
    if n_features < 6:
        raise ValueError("n_features must be at least 6 to support the default nonlinear signal")

    rng = np.random.default_rng(seed)
    train = _build_split(
        n_samples=n_train,
        n_features=n_features,
        base_noise=base_noise,
        hetero_strength=hetero_strength,
        rng=rng,
        split_name="train",
    )
    valid = _build_split(
        n_samples=n_valid,
        n_features=n_features,
        base_noise=base_noise,
        hetero_strength=hetero_strength,
        rng=rng,
        split_name="valid",
    )
    calib = _build_split(
        n_samples=n_calib,
        n_features=n_features,
        base_noise=base_noise,
        hetero_strength=hetero_strength,
        rng=rng,
        split_name="calib",
    )
    test = _build_split(
        n_samples=n_test,
        n_features=n_features,
        base_noise=base_noise,
        hetero_strength=hetero_strength,
        rng=rng,
        split_name="test",
    )

    feature_names = [f"x{j}" for j in range(n_features)]
    metadata = {
        "seed": int(seed),
        "base_noise": float(base_noise),
        "hetero_strength": float(hetero_strength),
        "split_scheme": "train/valid/calib/test",
    }
    return HeteroscedasticRegressionDataset(
        dataset_name="heteroscedastic_regression",
        train=train,
        valid=valid,
        calib=calib,
        test=test,
        feature_names=feature_names,
        metadata=metadata,
    )
