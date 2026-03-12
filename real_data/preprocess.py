from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .catalog import get_real_dataset_spec, list_real_dataset_names
from .download import download_real_dataset, get_cached_raw_frame
from .schema import (
    DEFAULT_REAL_DATA_ROOT,
    DEFAULT_REAL_PROCESSED_ROOT,
    dataset_processed_paths,
    dataset_raw_paths,
    ensure_parent_dirs,
    jsonable_mapping,
)


_MISSING_TOKENS = {"", "?", "na", "n/a", "none", "null", "nan"}
_PROCESSED_SAMPLE_ROWS = 5
_PROCESSED_FRAME_CACHE: Dict[tuple[str, str, str], tuple[pd.DataFrame, Dict[str, object]]] = {}


def _processed_cache_key(dataset_name: str, raw_root: Path, output_root: Path) -> tuple[str, str, str]:
    return str(dataset_name), str(Path(raw_root).resolve()), str(Path(output_root).resolve())

def _read_json_if_exists(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

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
    cached = get_cached_raw_frame(dataset_name=dataset_name, output_root=raw_root)
    if cached is not None:
        return cached
    spec = get_real_dataset_spec(dataset_name)
    raw_paths = dataset_raw_paths(dataset_name=dataset_name, root=raw_root)
    if spec.source_type == "openml":
        metadata = _read_json_if_exists(raw_paths.metadata_path)
        raw_table_is_sample = bool(metadata.get("raw_table_is_sample", False))
        if raw_table_is_sample or raw_paths.raw_table_path is None or not raw_paths.raw_table_path.exists():
            download_real_dataset(dataset_name=dataset_name, output_root=raw_root, overwrite=False)
            cached = get_cached_raw_frame(dataset_name=dataset_name, output_root=raw_root)
            if cached is not None:
                return cached
            metadata = _read_json_if_exists(raw_paths.metadata_path)
            raw_table_is_sample = bool(metadata.get("raw_table_is_sample", False))
        if raw_paths.raw_table_path is None or not raw_paths.raw_table_path.exists():
            raise FileNotFoundError(f"Raw table not found for {dataset_name!r}: {raw_paths.raw_table_path}")
        if raw_table_is_sample:
            raise RuntimeError(
                f"Raw table for {dataset_name!r} is stored only as a sample preview; full raw data was not materialized."
            )
        return pd.read_csv(raw_paths.raw_table_path, low_memory=False)

    if spec.source_type == "uci_archive":
        metadata = _read_json_if_exists(raw_paths.metadata_path)
        if not raw_paths.extracted_dir.exists() or len(list(raw_paths.extracted_dir.rglob("*"))) == 0:
            download_real_dataset(dataset_name=dataset_name, output_root=raw_root, overwrite=False)
            metadata = _read_json_if_exists(raw_paths.metadata_path)

        extracted_members = [str(member) for member in metadata.get("extracted_members", [])]
        member_path = _resolve_archive_member_path(
            dataset_name=dataset_name,
            raw_paths=raw_paths,
            archive_member=spec.archive_member,
            extracted_members=extracted_members,
        )
        member_path = _resolve_nested_archive_member(dataset_name=dataset_name, member_path=member_path)
        if member_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(member_path)
        if member_path.suffix.lower() == ".csv":
            return pd.read_csv(member_path, low_memory=False)
        raise ValueError(f"Unsupported archive member suffix for {dataset_name!r}: {member_path.suffix}")

    raise ValueError(f"Unsupported source_type for {dataset_name!r}: {spec.source_type}")





def _resolve_archive_member_path(*, dataset_name: str, raw_paths, archive_member: str | None, extracted_members: list[str]) -> Path:
    candidates: list[Path] = []
    if archive_member:
        candidates.append(raw_paths.extracted_dir / archive_member)

    normalized_requested = str(Path(archive_member).name).strip().lower() if archive_member else ""
    normalized_stem = str(Path(archive_member).stem).strip().lower() if archive_member else ""

    for member in extracted_members:
        member_path = raw_paths.extracted_dir / member
        if member_path.is_dir():
            continue
        member_name = member_path.name.strip().lower()
        member_stem = member_path.stem.strip().lower()
        if normalized_requested and member_name == normalized_requested:
            candidates.append(member_path)
        elif normalized_stem and member_stem == normalized_stem:
            candidates.append(member_path)

    filesystem_members = sorted(path for path in raw_paths.extracted_dir.rglob("*") if path.is_file())
    if len(candidates) == 0 and normalized_requested:
        basename_matches = [path for path in filesystem_members if path.name.strip().lower() == normalized_requested]
        candidates.extend(basename_matches)
    if len(candidates) == 0 and normalized_stem:
        stem_matches = [path for path in filesystem_members if path.stem.strip().lower() == normalized_stem]
        candidates.extend(stem_matches)
    if len(candidates) == 0:
        tabular_files = [path for path in filesystem_members if path.suffix.lower() in {".csv", ".xlsx", ".xls"}]
        if len(tabular_files) == 1:
            candidates.extend(tabular_files)

    unique_candidates: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(path)

    existing = [path for path in unique_candidates if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple archive members matched for {dataset_name!r}: {[str(path) for path in existing]}"
        )

    requested = archive_member if archive_member else "<unspecified>"
    raise FileNotFoundError(
        f"Archive member not found for {dataset_name!r}: requested {requested}; "
        f"available extracted files: {[str(path.relative_to(raw_paths.extracted_dir)) for path in filesystem_members]}"
    )


def _extract_nested_zip(member_path: Path) -> Path:
    target_dir = member_path.with_suffix("")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(member_path, mode="r") as zf:
        zf.extractall(target_dir)
    return target_dir


def _resolve_nested_archive_member(dataset_name: str, member_path: Path) -> Path:
    current = member_path
    seen: set[Path] = set()
    while current.suffix.lower() == ".zip":
        resolved = current.resolve()
        if resolved in seen:
            raise RuntimeError(f"Nested archive resolution loop detected for {dataset_name!r}: {current}")
        seen.add(resolved)
        extracted_dir = _extract_nested_zip(current)
        tabular_files = sorted(path for path in extracted_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"})
        if len(tabular_files) == 1:
            current = tabular_files[0]
            continue
        if len(tabular_files) == 0:
            nested_archives = sorted(path for path in extracted_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".zip")
            if len(nested_archives) == 1:
                current = nested_archives[0]
                continue
        raise ValueError(
            f"Unsupported nested archive layout for {dataset_name!r}: {current}; "
            f"tabular candidates={ [str(path.relative_to(extracted_dir)) for path in tabular_files] }"
        )
    return current


def _resolve_column_name(frame: pd.DataFrame, requested: str) -> str:
    requested_stripped = str(requested).strip()
    if requested_stripped in frame.columns:
        return requested_stripped

    lowered = {str(col).strip().lower(): str(col) for col in frame.columns}
    key = requested_stripped.lower()
    if key in lowered:
        return lowered[key]

    raise KeyError(
        f"Column {requested!r} not found after normalization. Available columns: "
        f"{frame.columns.astype(str).tolist()}"
    )



def _binary_target_from_series(
    series: pd.Series,
    positive_label: str,
    positive_aliases: tuple[str, ...],
) -> tuple[pd.Series, Dict[str, object]]:
    normalized = series.map(_normalize_label)
    observed = sorted(v for v in pd.unique(normalized) if v != "")
    if len(observed) != 2:
        raise ValueError(
            "Binary classification preprocessing expects exactly two non-missing target labels; "
            f"observed labels were: {observed}"
        )

    alias_set = {_normalize_label(positive_label), *(_normalize_label(v) for v in positive_aliases)}
    alias_set.discard("")
    matched_positive = sorted(label for label in observed if label in alias_set)

    if len(matched_positive) == 0:
        raise ValueError(
            "Could not map positive label "
            f"{positive_label!r}; observed target labels were: {observed}"
        )
    if len(matched_positive) > 1:
        raise ValueError(
            "Positive-label aliases are ambiguous after normalization; they matched multiple observed "
            f"target labels {matched_positive}. Check the dataset spec."
        )

    observed_positive_label = matched_positive[0]
    observed_negative_label = next(label for label in observed if label != observed_positive_label)
    target = (normalized == observed_positive_label).astype(int)
    return target, {
        "observed_labels": observed,
        "observed_positive_label": observed_positive_label,
        "observed_negative_label": observed_negative_label,
    }



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

    resolved_target_column = _resolve_column_name(clean, spec.target_column)
    clean = clean.dropna(subset=[resolved_target_column]).reset_index(drop=True)
    target, target_mapping = _binary_target_from_series(
        clean[resolved_target_column],
        positive_label=spec.positive_label,
        positive_aliases=spec.positive_aliases,
    )
    features = clean.drop(columns=[resolved_target_column]).copy()

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
        "resolved_raw_target_column": resolved_target_column,
        "processed_target_column": "__target__",
        "sample_id_column": "__sample_id__",
        "positive_label": spec.positive_label,
        "positive_aliases": list(spec.positive_aliases),
        "target_mapping": target_mapping,
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
        "stored_cleaned_sample_rows": int(min(_PROCESSED_SAMPLE_ROWS, processed.shape[0])),
        "cleaned_table_is_sample": True,
        "notes": spec.notes,
    }
    return processed, manifest


def materialize_real_dataset(
    dataset_name: str,
    raw_root: Path | str = DEFAULT_REAL_DATA_ROOT,
    output_root: Path | str = DEFAULT_REAL_PROCESSED_ROOT,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    cache_key = _processed_cache_key(dataset_name=dataset_name, raw_root=raw_root, output_root=output_root)
    cached = _PROCESSED_FRAME_CACHE.get(cache_key)
    if cached is not None:
        frame, manifest = cached
        return frame.copy(), dict(manifest)

    processed_paths = dataset_processed_paths(dataset_name=dataset_name, root=output_root)
    manifest = _read_json_if_exists(processed_paths.manifest_path)
    if processed_paths.cleaned_table_full_path.exists() and manifest:
        processed_frame = pd.read_csv(processed_paths.cleaned_table_full_path, low_memory=False)
        _PROCESSED_FRAME_CACHE[cache_key] = (processed_frame.copy(), dict(manifest))
        return processed_frame, dict(manifest)

    raw_frame = _load_raw_frame(dataset_name=dataset_name, raw_root=raw_root)
    processed_frame, manifest = _processed_table_from_raw(dataset_name=dataset_name, raw_frame=raw_frame)
    _PROCESSED_FRAME_CACHE[cache_key] = (processed_frame.copy(), dict(manifest))
    return processed_frame, manifest


def prepare_real_dataset(
    dataset_name: str,
    raw_root: Path | str = DEFAULT_REAL_DATA_ROOT,
    output_root: Path | str = DEFAULT_REAL_PROCESSED_ROOT,
    persist_full_table: bool = False,
) -> Path:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    processed_paths = dataset_processed_paths(dataset_name=dataset_name, root=output_root)
    ensure_parent_dirs([
        processed_paths.cleaned_table_path,
        processed_paths.cleaned_table_full_path,
        processed_paths.manifest_path,
    ])
    processed_paths.dataset_root.mkdir(parents=True, exist_ok=True)

    processed_frame, manifest = materialize_real_dataset(dataset_name=dataset_name, raw_root=raw_root, output_root=output_root)

    processed_frame.head(_PROCESSED_SAMPLE_ROWS).to_csv(processed_paths.cleaned_table_path, index=False)
    if bool(persist_full_table):
        processed_frame.to_csv(processed_paths.cleaned_table_full_path, index=False)
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
