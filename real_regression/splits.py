from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .catalog import get_real_regression_dataset_spec, list_real_regression_dataset_names
from .preprocess import materialize_real_regression_dataset
from .schema import (
    DEFAULT_REAL_REGRESSION_DATA_ROOT,
    DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
    dataset_split_paths,
    jsonable_mapping,
)


Array = np.ndarray


def _read_cleaned_table(dataset_name: str, processed_root: Path) -> pd.DataFrame:
    frame, _ = materialize_real_regression_dataset(dataset_name=dataset_name, raw_root=DEFAULT_REAL_REGRESSION_DATA_ROOT, output_root=processed_root)
    return frame


def _quantile_bin_labels(y: Array, max_bins: int = 10, max_classes: Optional[int] = None) -> Optional[Array]:
    y_arr = np.asarray(y, dtype=float)
    n = y_arr.shape[0]
    upper = max(2, min(int(max_bins), n))
    if max_classes is not None:
        upper = min(upper, int(max_classes))
    for q in range(upper, 1, -1):
        try:
            bins = pd.qcut(y_arr, q=q, duplicates="drop")
        except ValueError:
            continue
        labels = np.asarray(bins.astype(str), dtype=object)
        counts = pd.Series(labels).value_counts(dropna=False)
        if counts.shape[0] > 1 and int(counts.min()) >= 2 and counts.shape[0] <= q:
            return labels
    return None


def _split_indices(
    y: Array,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[Array, Array, Array]:
    total = float(train_ratio) + float(valid_ratio) + float(test_ratio)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0; got {total}")
    if min(train_ratio, valid_ratio, test_ratio) <= 0.0:
        raise ValueError("All split ratios must be positive")

    indices = np.arange(y.shape[0], dtype=int)
    n_test = max(1, int(np.ceil(y.shape[0] * float(test_ratio))))
    stratify_all = _quantile_bin_labels(y, max_classes=n_test)
    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify_all,
    )

    valid_fraction_within_train_valid = valid_ratio / (1.0 - test_ratio)
    n_valid = max(1, int(np.ceil(train_valid_idx.shape[0] * valid_fraction_within_train_valid)))
    stratify_train_valid = _quantile_bin_labels(y[train_valid_idx], max_classes=n_valid)

    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_fraction_within_train_valid,
        random_state=seed + 1,
        stratify=stratify_train_valid,
    )
    return np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)


def create_real_regression_split_manifest(
    dataset_name: str,
    repeat_id: int,
    raw_root: Path | str = DEFAULT_REAL_REGRESSION_DATA_ROOT,
    processed_root: Path | str = DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    output_root: Path | str = DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> Path:
    spec = get_real_regression_dataset_spec(dataset_name)
    frame, _ = materialize_real_regression_dataset(dataset_name=dataset_name, raw_root=Path(raw_root), output_root=Path(processed_root))
    y = frame["__target__"].to_numpy(dtype=float)
    train_idx, valid_idx, test_idx = _split_indices(
        y=y,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    n_unique = np.unique(np.concatenate([train_idx, valid_idx, test_idx])).shape[0]
    if n_unique != frame.shape[0]:
        raise RuntimeError(
            f"Split manifest for {dataset_name!r} repeat {repeat_id} does not cover each sample exactly once"
        )

    sample_ids = frame["__sample_id__"].astype(str).to_numpy()
    split_paths = dataset_split_paths(dataset_name=dataset_name, root=output_root)
    split_paths.split_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_paths.split_dir / f"repeat_{int(repeat_id):02d}.json"

    manifest = {
        "dataset_name": dataset_name,
        "repeat_id": int(repeat_id),
        "seed": int(seed),
        "split_strategy": spec.default_split_strategy,
        "split_ratios": {
            "train": float(train_ratio),
            "valid": float(valid_ratio),
            "test": float(test_ratio),
        },
        "n_samples": int(frame.shape[0]),
        "target_summary": {
            "overall_mean": float(np.mean(y)),
            "train_mean": float(np.mean(y[train_idx])) if train_idx.size else None,
            "valid_mean": float(np.mean(y[valid_idx])) if valid_idx.size else None,
            "test_mean": float(np.mean(y[test_idx])) if test_idx.size else None,
            "overall_std": float(np.std(y, ddof=0)),
            "train_std": float(np.std(y[train_idx], ddof=0)) if train_idx.size else None,
            "valid_std": float(np.std(y[valid_idx], ddof=0)) if valid_idx.size else None,
            "test_std": float(np.std(y[test_idx], ddof=0)) if test_idx.size else None,
        },
        "train_idx": train_idx.astype(int).tolist(),
        "valid_idx": valid_idx.astype(int).tolist(),
        "test_idx": test_idx.astype(int).tolist(),
        "train_sample_ids": sample_ids[train_idx].tolist(),
        "valid_sample_ids": sample_ids[valid_idx].tolist(),
        "test_sample_ids": sample_ids[test_idx].tolist(),
    }
    manifest_path.write_text(json.dumps(jsonable_mapping(manifest), indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def create_real_regression_split_manifests(
    dataset_names: Optional[list[str]] = None,
    raw_root: Path | str = DEFAULT_REAL_REGRESSION_DATA_ROOT,
    processed_root: Path | str = DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    output_root: Path | str = DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_repeats: int = 5,
    base_seed: int = 0,
) -> Dict[str, list[Path]]:
    names = dataset_names or list_real_regression_dataset_names()
    out: Dict[str, list[Path]] = {}
    for dataset_name in names:
        manifest_paths: list[Path] = []
        for repeat_id in range(int(n_repeats)):
            manifest_path = create_real_regression_split_manifest(
                dataset_name=dataset_name,
                repeat_id=repeat_id,
                raw_root=raw_root,
                processed_root=processed_root,
                output_root=output_root,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
                seed=int(base_seed) + int(repeat_id),
            )
            manifest_paths.append(manifest_path)
        out[dataset_name] = manifest_paths
    return out
