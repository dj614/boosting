from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .catalog import list_real_dataset_names
from .schema import (
    DEFAULT_REAL_PROCESSED_ROOT,
    DEFAULT_REAL_SPLIT_ROOT,
    dataset_processed_paths,
    dataset_split_paths,
    jsonable_mapping,
)


def _load_cleaned_table(dataset_name: str, processed_root: Path) -> pd.DataFrame:
    processed_paths = dataset_processed_paths(dataset_name=dataset_name, root=processed_root)
    if not processed_paths.cleaned_table_path.exists():
        raise FileNotFoundError(
            f"Cleaned table not found for {dataset_name!r}: {processed_paths.cleaned_table_path}"
        )
    return pd.read_csv(processed_paths.cleaned_table_path, low_memory=False)


def _normalize_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    total = float(train_ratio + valid_ratio + test_ratio)
    if total <= 0:
        raise ValueError("train_ratio + valid_ratio + test_ratio must be positive")
    return train_ratio / total, valid_ratio / total, test_ratio / total


def _stratified_random_split(
    y: np.ndarray,
    rng: np.random.Generator,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ratio, valid_ratio, test_ratio = _normalize_ratios(train_ratio, valid_ratio, test_ratio)
    indices = np.arange(y.shape[0], dtype=int)

    train_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(y):
        label_idx = indices[y == label]
        label_idx = rng.permutation(label_idx)
        n_label = int(label_idx.shape[0])

        n_train = int(np.floor(n_label * train_ratio))
        n_valid = int(np.floor(n_label * valid_ratio))
        n_test = n_label - n_train - n_valid

        if n_label >= 3:
            if n_train == 0:
                n_train = 1
            if n_valid == 0:
                n_valid = 1
            n_test = n_label - n_train - n_valid
            if n_test <= 0:
                if n_train >= n_valid and n_train > 1:
                    n_train -= 1
                elif n_valid > 1:
                    n_valid -= 1
                n_test = n_label - n_train - n_valid

        train_parts.append(label_idx[:n_train])
        valid_parts.append(label_idx[n_train : n_train + n_valid])
        test_parts.append(label_idx[n_train + n_valid :])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
    valid_idx = np.concatenate(valid_parts) if valid_parts else np.array([], dtype=int)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    train_idx = rng.permutation(train_idx)
    valid_idx = rng.permutation(valid_idx)
    test_idx = rng.permutation(test_idx)
    return train_idx, valid_idx, test_idx


def _fallback_random_split(
    n_samples: int,
    rng: np.random.Generator,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ratio, valid_ratio, test_ratio = _normalize_ratios(train_ratio, valid_ratio, test_ratio)
    idx = rng.permutation(np.arange(n_samples, dtype=int))
    n_train = int(np.floor(n_samples * train_ratio))
    n_valid = int(np.floor(n_samples * valid_ratio))
    n_test = n_samples - n_train - n_valid

    if n_samples >= 3:
        if n_train == 0:
            n_train = 1
        if n_valid == 0:
            n_valid = 1
        n_test = n_samples - n_train - n_valid
        if n_test <= 0:
            if n_train >= n_valid and n_train > 1:
                n_train -= 1
            elif n_valid > 1:
                n_valid -= 1
            n_test = n_samples - n_train - n_valid

    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]
    return train_idx, valid_idx, test_idx


def _has_both_classes(y: np.ndarray, indices: np.ndarray) -> bool:
    if indices.size == 0:
        return False
    return np.unique(y[indices]).shape[0] >= 2


def create_real_data_split_manifest(
    dataset_name: str,
    repeat_id: int,
    processed_root: Path | str = DEFAULT_REAL_PROCESSED_ROOT,
    output_root: Path | str = DEFAULT_REAL_SPLIT_ROOT,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> Path:
    processed_root = Path(processed_root)
    output_root = Path(output_root)
    frame = _load_cleaned_table(dataset_name=dataset_name, processed_root=processed_root)

    if "__target__" not in frame.columns:
        raise KeyError(f"Cleaned table for {dataset_name!r} must contain '__target__'")

    y = frame["__target__"].to_numpy(dtype=int)
    rng = np.random.default_rng(int(seed))
    train_idx, valid_idx, test_idx = _stratified_random_split(
        y=y,
        rng=rng,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    if not (_has_both_classes(y, train_idx) and _has_both_classes(y, valid_idx) and _has_both_classes(y, test_idx)):
        train_idx, valid_idx, test_idx = _fallback_random_split(
            n_samples=frame.shape[0],
            rng=np.random.default_rng(int(seed)),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )

    assigned = np.concatenate([train_idx, valid_idx, test_idx])
    if np.unique(assigned).shape[0] != frame.shape[0]:
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
        "split_ratios": {
            "train": float(train_ratio),
            "valid": float(valid_ratio),
            "test": float(test_ratio),
        },
        "n_samples": int(frame.shape[0]),
        "class_balance": {
            "overall_positive_rate": float(np.mean(y)),
            "train_positive_rate": float(np.mean(y[train_idx])) if train_idx.size else None,
            "valid_positive_rate": float(np.mean(y[valid_idx])) if valid_idx.size else None,
            "test_positive_rate": float(np.mean(y[test_idx])) if test_idx.size else None,
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

def create_real_data_split_manifests(
    dataset_names: Optional[list[str]] = None,
    processed_root: Path | str = DEFAULT_REAL_PROCESSED_ROOT,
    output_root: Path | str = DEFAULT_REAL_SPLIT_ROOT,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_repeats: int = 5,
    base_seed: int = 0,
) -> Dict[str, list[Path]]:
    names = dataset_names or list_real_dataset_names()
    out: Dict[str, list[Path]] = {}
    for dataset_name in names:
        manifest_paths: list[Path] = []
        for repeat_id in range(int(n_repeats)):
            manifest_path = create_real_data_split_manifest(
                dataset_name=dataset_name,
                repeat_id=repeat_id,
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