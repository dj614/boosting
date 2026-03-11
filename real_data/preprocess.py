from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .catalog import get_real_dataset_spec, list_real_dataset_names
from .schema import (
    DEFAULT_REAL_DATA_ROOT,
    DEFAULT_REAL_PROCESSED_ROOT,
    dataset_processed_paths,
    dataset_raw_paths,
    ensure_parent_dirs,
    jsonable_mapping,
)


_MISSING_TOKENS = {"", "?", "na", "n/a", "none", "null", "nan"}


def _normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


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
    spec = get_real_dataset_spec(dataset_name)
    raw_paths = dataset_raw_paths(dataset_name=dataset_name, root=raw_root)
    if spec.source_type == "openml":
        if raw_paths.raw_table_path is None or not raw_paths.raw_table_path.exists():
            raise FileNotFoundError(f"Raw table not found for {dataset_name!r}: {raw_paths.raw_table_path}")
        return pd.read_csv(raw_paths.raw_table_path, low_memory=False)

    if spec.source_type == "uci_archive":
        if not spec.archive_member:
            raise ValueError(f"Dataset {dataset_name!r} does not define archive_member")
        member_path = raw_paths.extracted_dir / spec.archive_member
        if not member_path.exists():
            raise FileNotFoundError(f"Archive member not found for {dataset_name!r}: {member_path}")
        if member_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(member_path)
        if member_path.suffix.lower() == ".csv":
            return pd.read_csv(member_path, low_memory=False)
        raise ValueError(f"Unsupported archive member suffix for {dataset_name!r}: {member_path.suffix}")

    raise ValueError(f"Unsupported source_type for {dataset_name!r}: {spec.source_type}")


def _binary_target_from_series(series: pd.Series, positive_label: str, positive_aliases: tuple[str, ...]) -> pd.Series:
    normalized = series.map(_normalize_label)
    alias_set = {_normalize_label(positive_label), *(_normalize_label(v) for v in positive_aliases)}
    alias_set.discard("")
    positive_mask = normalized.isin(alias_set)
    negative_mask = ~positive_mask

    if int(positive_mask.sum()) == 0:
        observed = sorted(v for v in pd.unique(normalized) if v != "")
        raise ValueError(
            "Could not map positive label "
            f"{positive_label!r}; observed target labels were: {observed}"
        )
    if int(negative_mask.sum()) == 0:
        raise ValueError("Target column appears to contain only positive labels after normalization")
    return positive_mask.astype(int)


def _infer_feature_columns(frame: pd.DataFrame) -> Dict[str, list[str]]:
    numeric_cols = frame.select_dtypes(include=[np.number, "bool"]).columns.astype(str).tolist()
    categorical_cols = [str(col) for col in frame.columns if str(col) not in numeric_cols]
    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def _processed_table_from_raw(dataset_name: str, raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, object]]:
    spec = get_real_dataset_spec(dataset_name)
    clean = _strip_and_standardize_missing(raw_frame)

    if spec.target_column not in clean.columns:
        raise KeyError(
            f"Target column {spec.target_column!r} not found for dataset {dataset_name!r}. "
            f"Available columns: {clean.columns.astype(str).tolist()}"
        )

    clean = clean.dropna(subset=[spec.target_column]).reset_index(drop=True)
    target = _binary_target_from_series(
        clean[spec.target_column],
        positive_label=spec.positive_label,
        positive_aliases=spec.positive_aliases,
    )
    features = clean.drop(columns=[spec.target_column]).copy()

    processed = pd.DataFrame(
        {
            "__sample_id__": [f"{dataset_name}_{i:06d}" for i in range(features.shape[0])],
            "__target__": target.astype(int).to_numpy(),
        }
    )
    for col in features.columns:
        processed[str(col)] = features[col]

    feature_info = _infer_feature_columns(features)
    manifest = {
        "dataset_name": spec.canonical_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "raw_target_column": spec.target_column,
        "processed_target_column": "__target__",
        "sample_id_column": "__sample_id__",
        "positive_label": spec.positive_label,
        "positive_aliases": list(spec.positive_aliases),
        "default_group_rules": list(spec.default_group_rules),
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
        "class_balance": {
            "negative_rate": float(1.0 - processed["__target__"].mean()),
            "positive_rate": float(processed["__target__"].mean()),
        },
        "notes": spec.notes,
    }
    return processed, manifest


def prepare_real_dataset(
    dataset_name: str,
    raw_root: Path | str = DEFAULT_REAL_DATA_ROOT,
    output_root: Path | str = DEFAULT_REAL_PROCESSED_ROOT,
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


def prepare_real_datasets(
    dataset_names: Optional[list[str]] = None,
    raw_root: Path | str = DEFAULT_REAL_DATA_ROOT,
    output_root: Path | str = DEFAULT_REAL_PROCESSED_ROOT,
) -> Dict[str, Path]:
    names = dataset_names or list_real_dataset_names()
    return {
        name: prepare_real_dataset(dataset_name=name, raw_root=raw_root, output_root=output_root)
        for name in names
    }