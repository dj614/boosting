from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from data.types import GroupedPartialLinearDataset, GroupedPartialLinearSplit

Array = np.ndarray


def _g_function(X: Array) -> Array:
    return (
        1.2 * np.sin(np.pi * X[:, 0])
        + 0.9 * X[:, 1] * X[:, 2]
        + 0.6 * (X[:, 3] ** 2)
        - 0.5 * np.abs(X[:, 4])
    ).astype(float)


def _make_correlated_error(
    n_groups: int,
    group_size: int,
    rho: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> Array:
    group_component = rng.normal(loc=0.0, scale=noise_scale, size=n_groups)
    idiosyncratic = rng.normal(loc=0.0, scale=noise_scale, size=n_groups * group_size)
    eta = np.repeat(np.sqrt(max(rho, 0.0)) * group_component, group_size)
    eta += np.sqrt(max(1.0 - rho, 0.0)) * idiosyncratic
    return eta.astype(float)


def _group_split_indices(
    n_groups: int,
    valid_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> Tuple[Array, Array, Array]:
    if valid_frac <= 0.0 or test_frac <= 0.0 or valid_frac + test_frac >= 1.0:
        raise ValueError("valid_frac and test_frac must be > 0 and sum to < 1")

    group_ids = np.arange(n_groups, dtype=int)
    rng.shuffle(group_ids)
    n_test = max(1, int(round(n_groups * test_frac)))
    n_valid = max(1, int(round(n_groups * valid_frac)))
    if n_test + n_valid >= n_groups:
        raise ValueError("Too few groups for the requested valid/test fractions")

    test_groups = np.sort(group_ids[:n_test])
    valid_groups = np.sort(group_ids[n_test : n_test + n_valid])
    train_groups = np.sort(group_ids[n_test + n_valid :])
    return train_groups, valid_groups, test_groups


def _slice_split(
    *,
    X: Array,
    D: Array,
    y: Array,
    group_id: Array,
    sample_id: Array,
    g_true: Array,
    group_effect_true: Array,
    eta_true: Array,
    selected_groups: Array,
    split_name: str,
) -> GroupedPartialLinearSplit:
    mask = np.isin(group_id, selected_groups)
    local_group = group_id[mask]
    metadata: Dict[str, Array] = {
        "group_size": np.full(mask.sum(), int(np.bincount(local_group).max()), dtype=int),
        "split_id": np.full(mask.sum(), split_name, dtype=object),
    }
    return GroupedPartialLinearSplit(
        X=X[mask].astype(float),
        D=D[mask].astype(float),
        y=y[mask].astype(float),
        group_id=local_group.astype(int),
        sample_id=sample_id[mask].astype(object),
        g_true=g_true[mask].astype(float),
        group_effect_true=group_effect_true[mask].astype(float),
        eta_true=eta_true[mask].astype(float),
        metadata=metadata,
    )


def generate_grouped_partial_linear_dataset(
    n_groups: int = 120,
    group_size: int = 8,
    n_features: int = 6,
    beta_true: float = 1.0,
    group_effect_scale: float = 0.75,
    noise_scale: float = 0.75,
    within_group_corr: float = 0.4,
    valid_group_frac: float = 0.2,
    test_group_frac: float = 0.2,
    seed: int = 0,
) -> GroupedPartialLinearDataset:
    if n_features < 5:
        raise ValueError("n_features must be at least 5 to support the default nuisance function")

    rng = np.random.default_rng(seed)
    n_obs = n_groups * group_size
    X = rng.normal(loc=0.0, scale=1.0, size=(n_obs, n_features)).astype(float)
    group_id = np.repeat(np.arange(n_groups, dtype=int), group_size)
    group_effect = rng.normal(loc=0.0, scale=group_effect_scale, size=n_groups)
    group_effect_obs = np.repeat(group_effect, group_size).astype(float)

    treatment_noise = rng.normal(loc=0.0, scale=0.8, size=n_obs)
    D = (
        0.7 * X[:, 0]
        - 0.5 * np.sin(np.pi * X[:, 1])
        + 0.3 * X[:, 2] * X[:, 3]
        + 0.15 * group_effect_obs
        + treatment_noise
    ).astype(float)

    g_true = _g_function(X)
    eta_true = _make_correlated_error(
        n_groups=n_groups,
        group_size=group_size,
        rho=within_group_corr,
        noise_scale=noise_scale,
        rng=rng,
    )
    y = (beta_true * D + g_true + group_effect_obs + eta_true).astype(float)

    sample_id = np.asarray([f"obs_{i}" for i in range(n_obs)], dtype=object)
    train_groups, valid_groups, test_groups = _group_split_indices(
        n_groups=n_groups,
        valid_frac=valid_group_frac,
        test_frac=test_group_frac,
        rng=rng,
    )

    train = _slice_split(
        X=X,
        D=D,
        y=y,
        group_id=group_id,
        sample_id=sample_id,
        g_true=g_true,
        group_effect_true=group_effect_obs,
        eta_true=eta_true,
        selected_groups=train_groups,
        split_name="train",
    )
    valid = _slice_split(
        X=X,
        D=D,
        y=y,
        group_id=group_id,
        sample_id=sample_id,
        g_true=g_true,
        group_effect_true=group_effect_obs,
        eta_true=eta_true,
        selected_groups=valid_groups,
        split_name="valid",
    )
    test = _slice_split(
        X=X,
        D=D,
        y=y,
        group_id=group_id,
        sample_id=sample_id,
        g_true=g_true,
        group_effect_true=group_effect_obs,
        eta_true=eta_true,
        selected_groups=test_groups,
        split_name="test",
    )

    feature_names: List[str] = [f"x{j}" for j in range(n_features)]
    metadata = {
        "seed": int(seed),
        "n_groups": int(n_groups),
        "group_size": int(group_size),
        "within_group_corr": float(within_group_corr),
        "split_scheme": "group-wise train/valid/test",
    }
    return GroupedPartialLinearDataset(
        dataset_name="grouped_partial_linear",
        train=train,
        valid=valid,
        test=test,
        beta_true=float(beta_true),
        feature_names=feature_names,
        metadata=metadata,
    )
