from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .catalog import get_real_regression_dataset_spec, list_real_regression_dataset_names
from .schema import (
    DEFAULT_REAL_REGRESSION_DATA_ROOT,
    DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    dataset_processed_paths,
    dataset_raw_paths,
    ensure_parent_dirs,
    jsonable_mapping,
)


_MISSING_TOKENS = {"", "?", "na", "n/a", "none", "null", "nan"}


def _strip_and_standardize_missing(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    clean.columns = [str(col).strip() for col in clean.columns]
    for col in clean.columns:
        series = clean[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            normalized = series.astype("string").str.strip()
            normalized = normalized.mask(normalized.str.lower().isin(_MISSING_TOKENS), other=pd.NA)
            clean[col] = normalized
    return clean


def _load_raw_frame(dataset_name: str, raw_root: Path) -> pd.DataFrame:
    raw_paths = dataset_raw_paths(dataset_name=dataset_name, root=raw_root)
    if raw_paths.raw_table_path is None or not raw_paths.raw_table_path.exists():
        raise FileNotFoundError(f"Raw table not found for {dataset_name!r}: {raw_paths.raw_table_path}")
    return pd.read_csv(raw_paths.raw_table_path, low_memory=False)


def _infer_feature_columns(frame: pd.DataFrame, categorical_hint: tuple[str, ...]) -> Dict[str, list[str]]:
    numeric_cols = frame.select_dtypes(include=[np.number, "bool"]).columns.astype(str).tolist()
    categorical_hint_set = {str(col) for col in categorical_hint}
    categorical_cols = [
        str(col)
        for col in frame.columns
        if str(col) not in numeric_cols or str(col) in categorical_hint_set
    ]
    numeric_cols = [str(col) for col in frame.columns if str(col) not in categorical_cols]
    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def _processed_table_from_raw(dataset_name: str, raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, object]]:
    spec = get_real_regression_dataset_spec(dataset_name)
    clean = _strip_and_standardize_missing(raw_frame)

    if spec.target_column not in clean.columns:
        raise KeyError(
            f"Target column {spec.target_column!r} not found for dataset {dataset_name!r}. "
            f"Available columns: {clean.columns.astype(str).tolist()}"
        )

    feature_columns = list(spec.feature_columns) if spec.feature_columns else [
        str(col) for col in clean.columns if str(col) != spec.target_column
    ]
    missing_features = [col for col in feature_columns if col not in clean.columns]
    if missing_features:
        raise KeyError(
            f"Feature columns missing for dataset {dataset_name!r}: {missing_features}. "
            f"Available columns: {clean.columns.astype(str).tolist()}"
        )

    clean = clean.dropna(subset=[spec.target_column]).reset_index(drop=True)
    target = pd.to_numeric(clean[spec.target_column], errors="coerce")
    keep_mask = target.notna().to_numpy()
    clean = clean.loc[keep_mask].reset_index(drop=True)
    target = target.loc[keep_mask].reset_index(drop=True)
    features = clean.loc[:, feature_columns].copy()

    processed = pd.DataFrame(
        {
            "__sample_id__": [f"{dataset_name}_{i:06d}" for i in range(features.shape[0])],
            "__target__": target.astype(float).to_numpy(),
        }
    )
    for col in features.columns:
        processed[str(col)] = features[col]

    feature_info = _infer_feature_columns(features, categorical_hint=spec.categorical_columns)
    manifest = {
        "dataset_name": spec.canonical_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "raw_target_column": spec.target_column,
        "processed_target_column": "__target__",
        "sample_id_column": "__sample_id__",
        "default_split_strategy": spec.default_split_strategy,
        "n_rows": int(processed.shape[0]),
        "n_columns": int(processed.shape[1]),
        "n_feature_columns": int(features.shape[1]),
        "feature_columns": features.columns.astype(str).tolist(),
        "numeric_columns": feature_info["numeric_columns"],
        "categorical_columns": feature_info["categorical_columns"],
        "missing_per_column": {
            str(col): int(count)
            for col, count in processed.isna().sum().sort_index().items()
            if int(count) > 0
        },
        "target_summary": {
            "mean": float(processed["__target__"].mean()),
            "std": float(processed["__target__"].std(ddof=0)),
            "min": float(processed["__target__"].min()),
            "max": float(processed["__target__"].max()),
        },
        "notes": spec.notes,
    }
    return processed, manifest


def prepare_real_regression_dataset(
    dataset_name: str,
    raw_root: Path | str = DEFAULT_REAL_REGRESSION_DATA_ROOT,
    output_root: Path | str = DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
) -> Path:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    processed_paths = dataset_processed_paths(dataset_name=dataset_name, root=output_root)
    ensure_parent_dirs([processed_paths.cleaned_table_path, processed_paths.manifest_path])
    processed_paths.dataset_root.mkdir(parents=True, exist_ok=True)

    raw_frame = _load_raw_frame(dataset_name=dataset_name, raw_root=raw_root)
    processed_frame, manifest = _processed_table_from_raw(dataset_name=dataset_name, raw_frame=raw_frame)

    processed_frame.to_csv(processed_paths.cleaned_table_path, index=False)
    processed_paths.manifest_path.write_text(
        json.dumps(jsonable_mapping(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return processed_paths.dataset_root


def prepare_real_regression_datasets(
    dataset_names: Optional[list[str]] = None,
    raw_root: Path | str = DEFAULT_REAL_REGRESSION_DATA_ROOT,
    output_root: Path | str = DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
) -> Dict[str, Path]:
    names = dataset_names or list_real_regression_dataset_names()
