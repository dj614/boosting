from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd

from data.types import TabularRegressionDataset, TabularRegressionSplit
from sim.grouped_classification_data import _build_preprocessor, _feature_names_from_preprocessor

from .schema import (
    DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
    dataset_processed_paths,
    dataset_split_paths,
)


Array = np.ndarray


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_metadata(metadata: Mapping[str, Array], index: Array) -> Dict[str, Array]:
    out: Dict[str, Array] = {}
    for key, value in metadata.items():
        out[key] = np.asarray(value)[index]
    return out


def _make_split(
    X: Array,
    y: Array,
    sample_id: Array,
    index: Array,
    metadata: Mapping[str, Array],
) -> TabularRegressionSplit:
    return TabularRegressionSplit(
        X=np.asarray(X[index], dtype=float),
        y=np.asarray(y[index], dtype=float),
        sample_id=np.asarray(sample_id[index], dtype=object),
        metadata=_slice_metadata(metadata, index),
    )


def _load_cleaned_table(dataset_name: str, processed_root: Path) -> tuple[pd.DataFrame, Dict[str, object]]:
    processed_paths = dataset_processed_paths(dataset_name=dataset_name, root=processed_root)
    if not processed_paths.cleaned_table_path.exists():
        raise FileNotFoundError(
            f"Cleaned table not found for {dataset_name!r}: {processed_paths.cleaned_table_path}"
        )
    if not processed_paths.manifest_path.exists():
        raise FileNotFoundError(
            f"Processed manifest not found for {dataset_name!r}: {processed_paths.manifest_path}"
        )
    frame = pd.read_csv(processed_paths.cleaned_table_path, low_memory=False)
    manifest = _read_json(processed_paths.manifest_path)
    return frame, manifest


def _load_split_manifest(dataset_name: str, repeat_id: int, split_root: Path) -> Dict[str, object]:
    split_paths = dataset_split_paths(dataset_name=dataset_name, root=split_root)
    manifest_path = split_paths.split_dir / f"repeat_{int(repeat_id):02d}.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found for {dataset_name!r}: {manifest_path}")
    return _read_json(manifest_path)


def load_real_regression_dataset(
    dataset_name: str,
    repeat_id: int = 0,
    processed_root: Path | str = DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    split_root: Path | str = DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
) -> TabularRegressionDataset:
    processed_root = Path(processed_root)
    split_root = Path(split_root)
    frame, processed_manifest = _load_cleaned_table(dataset_name=dataset_name, processed_root=processed_root)
    split_manifest = _load_split_manifest(dataset_name=dataset_name, repeat_id=repeat_id, split_root=split_root)

    train_idx = np.asarray(split_manifest["train_idx"], dtype=int)
    valid_idx = np.asarray(split_manifest["valid_idx"], dtype=int)
    test_idx = np.asarray(split_manifest["test_idx"], dtype=int)
    n_total = frame.shape[0]
    all_idx = np.sort(np.concatenate([train_idx, valid_idx, test_idx]))
    if all_idx.shape[0] != n_total or not np.array_equal(all_idx, np.arange(n_total, dtype=int)):
        raise RuntimeError(
            f"Split manifest for {dataset_name!r} repeat {repeat_id} does not form a partition of all rows"
        )

    sample_id = frame["__sample_id__"].astype(str).to_numpy(dtype=object)
    y = frame["__target__"].to_numpy(dtype=float)
    raw_features = frame.drop(columns=["__sample_id__", "__target__"]).copy()

    preprocessor = _build_preprocessor(raw_features.iloc[train_idx])
    X_train = preprocessor.fit_transform(raw_features.iloc[train_idx])
    X_valid = preprocessor.transform(raw_features.iloc[valid_idx])
    X_test = preprocessor.transform(raw_features.iloc[test_idx])

    X_all = np.zeros((raw_features.shape[0], X_train.shape[1]), dtype=float)
    X_all[train_idx] = np.asarray(X_train, dtype=float)
    X_all[valid_idx] = np.asarray(X_valid, dtype=float)
    X_all[test_idx] = np.asarray(X_test, dtype=float)

    metadata_arrays = {
        "row_index": np.arange(frame.shape[0], dtype=int),
    }
    metadata = {
        "preprocessor": preprocessor,
        "raw_feature_columns": raw_features.columns.astype(str).tolist(),
        "split_sizes": {
            "train": int(train_idx.shape[0]),
            "valid": int(valid_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "processed_manifest": processed_manifest,
        "split_manifest": split_manifest,
    }

    return TabularRegressionDataset(
        dataset_name=dataset_name,
        train=_make_split(X_all, y, sample_id, train_idx, metadata_arrays),
        valid=_make_split(X_all, y, sample_id, valid_idx, metadata_arrays),
        test=_make_split(X_all, y, sample_id, test_idx, metadata_arrays),
        feature_names=_feature_names_from_preprocessor(preprocessor),
        metadata=metadata,
    )
