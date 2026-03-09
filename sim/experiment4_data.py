from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

Array = np.ndarray
DesignType = Literal["independent", "block_correlated", "strong_collinear"]
BetaPatternType = Literal["equal", "decay", "mixed_sign"]


@dataclass(frozen=True)
class SparseRegressionSplit:
    X: Array
    y: Array
    signal: Array
    meta: Dict[str, Array]


@dataclass(frozen=True)
class SparseRegressionDataset:
    design: DesignType
    beta_pattern: BetaPatternType
    train: SparseRegressionSplit
    valid: SparseRegressionSplit
    test: SparseRegressionSplit
    beta_true: Array
    support_true: Array
    feature_names: Tuple[str, ...]
    config: Dict[str, object]



def make_active_coefficients(
    s: int,
    beta_scale: float = 1.0,
    beta_pattern: BetaPatternType = "equal",
) -> Array:
    if s <= 0:
        raise ValueError("s must be positive")

    if beta_pattern == "equal":
        coeffs = np.full(s, beta_scale, dtype=float)
    elif beta_pattern == "decay":
        coeffs = beta_scale * np.linspace(1.0, 0.4, num=s, dtype=float)
    elif beta_pattern == "mixed_sign":
        magnitudes = beta_scale * np.linspace(1.0, 0.6, num=s, dtype=float)
        signs = np.where(np.arange(s) % 2 == 0, 1.0, -1.0)
        coeffs = magnitudes * signs
    else:
        raise ValueError(f"Unsupported beta_pattern={beta_pattern!r}")
    return coeffs.astype(float)



def _make_block_correlation_block(block_len: int, rho: float) -> Array:
    idx = np.arange(block_len)
    return rho ** np.abs(idx[:, None] - idx[None, :])



def make_covariance_matrix(
    p: int,
    design: DesignType = "independent",
    rho: float = 0.5,
    block_size: int = 20,
    active_idx: Optional[Array] = None,
    collinear_cluster_size: int = 6,
    collinear_strength: float = 0.995,
) -> Array:
    if p <= 0:
        raise ValueError("p must be positive")
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must lie in [0, 1)")

    sigma = np.eye(p, dtype=float)

    if design == "independent":
        return sigma

    if design == "block_correlated":
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        sigma = np.zeros((p, p), dtype=float)
        start = 0
        while start < p:
            end = min(start + block_size, p)
            sigma[start:end, start:end] = _make_block_correlation_block(end - start, rho)
            start = end
        return sigma

    if design == "strong_collinear":
        sigma = np.eye(p, dtype=float)
        if active_idx is None:
            raise ValueError("active_idx must be provided for strong_collinear design")

        active_idx = np.asarray(active_idx, dtype=int)
        available = [j for j in range(p) if j not in set(active_idx.tolist())]
        cursor = 0
        cluster_partner_map = np.full((active_idx.shape[0], collinear_cluster_size), -1, dtype=int)

        for group_id, src in enumerate(active_idx):
            for slot in range(collinear_cluster_size):
                if cursor >= len(available):
                    break
                tgt = available[cursor]
                cursor += 1
                sigma[src, tgt] = collinear_strength
                sigma[tgt, src] = collinear_strength
                cluster_partner_map[group_id, slot] = tgt

            group_members = [src] + [x for x in cluster_partner_map[group_id].tolist() if x >= 0]
            for i in group_members:
                for j in group_members:
                    if i == j:
                        sigma[i, j] = 1.0
                    else:
                        sigma[i, j] = max(sigma[i, j], collinear_strength)

        # Add mild background correlation to make the rest of the design less trivial.
        sigma = 0.15 * np.ones((p, p), dtype=float) + 0.85 * sigma
        np.fill_diagonal(sigma, 1.0)
        return sigma

    raise ValueError(f"Unsupported design={design!r}")



def _matrix_square_root(sigma: Array) -> Array:
    jitter = 1e-10 * np.eye(sigma.shape[0], dtype=float)
    try:
        return np.linalg.cholesky(sigma + jitter).astype(float)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.clip(eigvals, 1e-10, None)
        return (eigvecs * np.sqrt(eigvals)[None, :]).astype(float)


def _sample_multivariate_gaussian(
    n_samples: int,
    factor: Array,
    rng: np.random.Generator,
) -> Array:
    z = rng.normal(size=(n_samples, factor.shape[0]))
    return (z @ factor.T).astype(float)



def _resolve_support_indices(
    p: int,
    s: int,
    support_strategy: Literal["first", "spaced", "random"],
    rng: np.random.Generator,
) -> Array:
    if s <= 0 or s > p:
        raise ValueError("s must satisfy 1 <= s <= p")

    if support_strategy == "first":
        return np.arange(s, dtype=int)
    if support_strategy == "spaced":
        return np.linspace(0, p - 1, num=s, dtype=int)
    if support_strategy == "random":
        return np.sort(rng.choice(p, size=s, replace=False).astype(int))
    raise ValueError(f"Unsupported support_strategy={support_strategy!r}")



def make_sparse_beta(
    p: int,
    s: int,
    beta_scale: float = 1.0,
    beta_pattern: BetaPatternType = "equal",
    support_strategy: Literal["first", "spaced", "random"] = "spaced",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Array, Array]:
    if rng is None:
        rng = np.random.default_rng()

    active_idx = _resolve_support_indices(p=p, s=s, support_strategy=support_strategy, rng=rng)
    beta = np.zeros(p, dtype=float)
    beta[active_idx] = make_active_coefficients(s=s, beta_scale=beta_scale, beta_pattern=beta_pattern)
    return beta, active_idx



def _noise_std_from_snr(signal: Array, snr: float) -> float:
    if snr <= 0.0:
        raise ValueError("snr must be positive")
    signal_var = float(np.var(signal))
    if signal_var <= 0.0:
        return 1e-8
    return float(np.sqrt(signal_var / snr))



def _standardize_from_train(train_X: Array, other_X: Array) -> Tuple[Array, Array, Array, Array]:
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (train_X - mean) / std, (other_X - mean) / std, mean.astype(float), std.astype(float)



def _build_split(
    X: Array,
    beta_true: Array,
    noise_std: float,
    rng: np.random.Generator,
    feature_mean: Array,
    feature_std: Array,
    design: DesignType,
    support_true: Array,
) -> SparseRegressionSplit:
    signal = X @ beta_true
    eps = rng.normal(loc=0.0, scale=noise_std, size=X.shape[0])
    y = signal + eps

    corr_to_signal = np.corrcoef(X, signal[:, None], rowvar=False)[-1, :-1]
    corr_to_signal = np.nan_to_num(corr_to_signal, nan=0.0)

    meta: Dict[str, Array] = {
        "feature_mean": feature_mean.astype(float),
        "feature_std": feature_std.astype(float),
        "signal": signal.astype(float),
        "noise_std": np.full(X.shape[0], noise_std, dtype=float),
        "active_mask": np.isin(np.arange(X.shape[1]), support_true).astype(int),
        "corr_to_signal": corr_to_signal.astype(float),
        "design_id": np.full(X.shape[0], ["independent", "block_correlated", "strong_collinear"].index(design), dtype=int),
    }
    return SparseRegressionSplit(X=X.astype(float), y=y.astype(float), signal=signal.astype(float), meta=meta)



def generate_sparse_regression_dataset(
    n_train: int = 200,
    n_valid: int = 200,
    n_test: int = 2000,
    p: int = 2000,
    s: int = 10,
    design: DesignType = "independent",
    rho: float = 0.5,
    block_size: int = 20,
    beta_scale: float = 1.0,
    beta_pattern: BetaPatternType = "equal",
    support_strategy: Literal["first", "spaced", "random"] = "spaced",
    snr: float = 4.0,
    seed: int = 0,
    standardize: bool = True,
    collinear_cluster_size: int = 6,
    collinear_strength: float = 0.995,
) -> SparseRegressionDataset:
    rng = np.random.default_rng(seed)

    beta_true, support_true = make_sparse_beta(
        p=p,
        s=s,
        beta_scale=beta_scale,
        beta_pattern=beta_pattern,
        support_strategy=support_strategy,
        rng=rng,
    )

    sigma = make_covariance_matrix(
        p=p,
        design=design,
        rho=rho,
        block_size=block_size,
        active_idx=support_true,
        collinear_cluster_size=collinear_cluster_size,
        collinear_strength=collinear_strength,
    )

    sigma_factor = _matrix_square_root(sigma)
    X_train_raw = _sample_multivariate_gaussian(n_samples=n_train, factor=sigma_factor, rng=rng)
    X_valid_raw = _sample_multivariate_gaussian(n_samples=n_valid, factor=sigma_factor, rng=rng)
    X_test_raw = _sample_multivariate_gaussian(n_samples=n_test, factor=sigma_factor, rng=rng)

    if standardize:
        X_train, X_valid, feature_mean, feature_std = _standardize_from_train(X_train_raw, X_valid_raw)
        _, X_test, _, _ = _standardize_from_train(X_train_raw, X_test_raw)
    else:
        X_train = X_train_raw
        X_valid = X_valid_raw
        X_test = X_test_raw
        feature_mean = np.zeros(p, dtype=float)
        feature_std = np.ones(p, dtype=float)

    signal_train = X_train @ beta_true
    noise_std = _noise_std_from_snr(signal=signal_train, snr=snr)

    train = _build_split(
        X=X_train,
        beta_true=beta_true,
        noise_std=noise_std,
        rng=rng,
        feature_mean=feature_mean,
        feature_std=feature_std,
        design=design,
        support_true=support_true,
    )
    valid = _build_split(
        X=X_valid,
        beta_true=beta_true,
        noise_std=noise_std,
        rng=rng,
        feature_mean=feature_mean,
        feature_std=feature_std,
        design=design,
        support_true=support_true,
    )
    test = _build_split(
        X=X_test,
        beta_true=beta_true,
        noise_std=noise_std,
        rng=rng,
        feature_mean=feature_mean,
        feature_std=feature_std,
        design=design,
        support_true=support_true,
    )

    feature_names = tuple(f"x{j}" for j in range(p))
    config: Dict[str, object] = {
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test,
        "p": p,
        "s": s,
        "design": design,
        "rho": rho,
        "block_size": block_size,
        "beta_scale": beta_scale,
        "beta_pattern": beta_pattern,
        "support_strategy": support_strategy,
        "snr": snr,
        "seed": seed,
        "standardize": standardize,
        "collinear_cluster_size": collinear_cluster_size,
        "collinear_strength": collinear_strength,
    }

    return SparseRegressionDataset(
        design=design,
        beta_pattern=beta_pattern,
        train=train,
        valid=valid,
        test=test,
        beta_true=beta_true.astype(float),
        support_true=support_true.astype(int),
        feature_names=feature_names,
        config=config,
    )



def summarize_sparse_regression_dataset(dataset: SparseRegressionDataset) -> Dict[str, float]:
    active_idx = dataset.support_true
    inactive_mask = np.ones(dataset.beta_true.shape[0], dtype=bool)
    inactive_mask[active_idx] = False

    train_signal = dataset.train.signal
    test_signal = dataset.test.signal
    train_X = dataset.train.X
    sigma_hat = np.corrcoef(train_X, rowvar=False)

    off_diag = sigma_hat[~np.eye(sigma_hat.shape[0], dtype=bool)]
    summary: Dict[str, float] = {
        "train_n": float(dataset.train.X.shape[0]),
        "valid_n": float(dataset.valid.X.shape[0]),
        "test_n": float(dataset.test.X.shape[0]),
        "p": float(dataset.train.X.shape[1]),
        "s": float(dataset.support_true.shape[0]),
        "p_over_n": float(dataset.train.X.shape[1] / dataset.train.X.shape[0]),
        "support_min": float(active_idx.min()),
        "support_max": float(active_idx.max()),
        "beta_l0": float(np.count_nonzero(dataset.beta_true)),
        "beta_l1": float(np.abs(dataset.beta_true).sum()),
        "beta_l2": float(np.linalg.norm(dataset.beta_true)),
        "train_signal_mean": float(train_signal.mean()),
        "train_signal_std": float(train_signal.std()),
        "test_signal_std": float(test_signal.std()),
        "train_y_std": float(dataset.train.y.std()),
        "noise_std": float(dataset.train.meta["noise_std"][0]),
        "empirical_train_snr": float(np.var(train_signal) / (dataset.train.meta["noise_std"][0] ** 2)),
        "active_abs_beta_mean": float(np.mean(np.abs(dataset.beta_true[active_idx]))),
        "inactive_abs_beta_mean": float(np.mean(np.abs(dataset.beta_true[inactive_mask]))),
        "train_feature_std_mean": float(dataset.train.X.std(axis=0).mean()),
        "train_feature_std_min": float(dataset.train.X.std(axis=0).min()),
        "sample_corr_offdiag_mean_abs": float(np.mean(np.abs(off_diag))),
        "sample_corr_offdiag_q95_abs": float(np.quantile(np.abs(off_diag), 0.95)),
    }
    return summary



def top_correlated_features(
    dataset: SparseRegressionDataset,
    top_k: int = 5,
) -> Dict[str, Array]:
    X = dataset.train.X
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    active_idx = dataset.support_true
    neighbor_idx = np.argsort(-np.abs(corr[active_idx]), axis=1)[:, :top_k]
    neighbor_corr = np.take_along_axis(corr[active_idx], neighbor_idx, axis=1)

    return {
        "active_idx": active_idx.astype(int),
        "neighbor_idx": neighbor_idx.astype(int),
        "neighbor_corr": neighbor_corr.astype(float),
    }
