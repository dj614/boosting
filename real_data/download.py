from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlopen

import pandas as pd

try:  # pragma: no cover - sklearn import is environment-dependent
    from sklearn.datasets import fetch_openml
except Exception:  # pragma: no cover
    fetch_openml = None

from .catalog import get_real_dataset_spec, list_real_dataset_names
from .schema import DEFAULT_REAL_DATA_ROOT, dataset_raw_paths, ensure_parent_dirs, jsonable_mapping

_RAW_SAMPLE_ROWS = 5
_RAW_FRAME_CACHE: Dict[tuple[str, str], pd.DataFrame] = {}


def _cache_key(dataset_name: str, output_root: Path) -> tuple[str, str]:
    return str(dataset_name), str(Path(output_root).resolve())


def get_cached_raw_frame(dataset_name: str, output_root: Path | str = DEFAULT_REAL_DATA_ROOT) -> Optional[pd.DataFrame]:
    frame = _RAW_FRAME_CACHE.get(_cache_key(dataset_name=dataset_name, output_root=Path(output_root)))
    if frame is None:
        return None
    return frame.copy()

def _download_bytes(url: str) -> bytes:
    with urlopen(url) as response:  # nosec - downloading known public datasets
        return response.read()

def _openml_features_to_frame(data, feature_names) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        matrix = data.toarray() if hasattr(data, "toarray") else data
        columns = [str(name).strip() for name in (feature_names or [])]
        frame = pd.DataFrame(matrix, columns=columns or None)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame.reset_index(drop=True)


def _openml_target_to_series(target, target_column: str, n_rows: int) -> pd.Series:
    if isinstance(target, pd.Series):
        series = target.copy()
    else:
        series = pd.Series(target)
    series = series.reset_index(drop=True)
    if int(series.shape[0]) != int(n_rows):
        raise ValueError(
            f"OpenML target length mismatch for target column {target_column!r}: "
            f"got {series.shape[0]} rows for {n_rows} feature rows"
        )
    series.name = str(target_column)
    return series

def _write_raw_table_sample(dataset_name: str, output_root: Path, frame: pd.DataFrame) -> Path:
    paths = dataset_raw_paths(dataset_name=dataset_name, root=output_root)
    ensure_parent_dirs([paths.raw_table_path])
    frame.head(_RAW_SAMPLE_ROWS).to_csv(paths.raw_table_path, index=False)
    _RAW_FRAME_CACHE[_cache_key(dataset_name=dataset_name, output_root=output_root)] = frame.copy()
    return paths.raw_table_path

def _openml_features_to_frame(data, feature_names) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        matrix = data.toarray() if hasattr(data, "toarray") else data
        columns = [str(name) for name in (feature_names or [])]
        frame = pd.DataFrame(matrix, columns=columns or None)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame.reset_index(drop=True)


def _openml_target_to_series(target, target_column: str, n_rows: int) -> pd.Series:
    if isinstance(target, pd.Series):
        series = target.copy()
    else:
        series = pd.Series(target)
    series = series.reset_index(drop=True)
    if int(series.shape[0]) != int(n_rows):
        raise ValueError(
            f"OpenML target length mismatch for target column {target_column!r}: "
            f"got {series.shape[0]} rows for {n_rows} feature rows"
        )
    series.name = str(target_column)
    return series

def _save_openml_table(dataset_name: str, output_root: Path) -> Dict[str, object]:
    if fetch_openml is None:  # pragma: no cover
        raise ImportError("scikit-learn fetch_openml is unavailable in this environment")

    spec = get_real_dataset_spec(dataset_name)
    bunch = fetch_openml(name=spec.openml_name, version=spec.openml_version, as_frame="auto")
    frame = _openml_features_to_frame(bunch.data, getattr(bunch, "feature_names", None))
    frame[spec.target_column] = _openml_target_to_series(
        bunch.target,
        target_column=spec.target_column,
        n_rows=frame.shape[0],
    )
    frame = bunch.data.copy()
    frame[spec.target_column] = pd.Series(bunch.target)

    paths = dataset_raw_paths(dataset_name=dataset_name, root=output_root)
    ensure_parent_dirs([paths.raw_table_path, paths.metadata_path])
    sample_path = _write_raw_table_sample(dataset_name=dataset_name, output_root=output_root, frame=frame)

    metadata = {
        "dataset_name": spec.canonical_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "target_column": spec.target_column,
        "positive_label": spec.positive_label,
        "openml_name": spec.openml_name,
        "openml_version": spec.openml_version,
        "raw_table_path": str(sample_path),
        "raw_table_is_sample": True,
        "raw_table_sample_rows": int(min(_RAW_SAMPLE_ROWS, frame.shape[0])),
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "raw_columns": frame.columns.astype(str).tolist(),
        "default_group_rules": list(spec.default_group_rules),
        "notes": spec.notes,
    }
    return metadata


def _save_uci_archive(dataset_name: str, output_root: Path) -> Dict[str, object]:
    spec = get_real_dataset_spec(dataset_name)
    if not spec.uci_download_url:
        raise ValueError(f"Dataset {dataset_name!r} does not define a UCI download URL")

    paths = dataset_raw_paths(dataset_name=dataset_name, root=output_root)
    archive_suffix = Path(spec.uci_download_url).suffix or ".zip"
    archive_path = paths.raw_archive_path.with_suffix(archive_suffix)
    ensure_parent_dirs([archive_path, paths.metadata_path])
    paths.extracted_dir.mkdir(parents=True, exist_ok=True)

    payload = _download_bytes(spec.uci_download_url)
    archive_path.write_bytes(payload)

    extracted_members = []
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(paths.extracted_dir)
            extracted_members = sorted(zf.namelist())
    else:  # pragma: no cover
        raise ValueError(f"Unsupported archive suffix for {dataset_name!r}: {archive_path.suffix}")

    metadata = {
        "dataset_name": spec.canonical_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "target_column": spec.target_column,
        "positive_label": spec.positive_label,
        "uci_download_url": spec.uci_download_url,
        "archive_member": spec.archive_member,
        "raw_archive_path": str(archive_path),
        "extracted_dir": str(paths.extracted_dir),
        "extracted_members": extracted_members,
        "default_group_rules": list(spec.default_group_rules),
        "notes": spec.notes,
    }
    return metadata


def download_real_dataset(
    dataset_name: str,
    output_root: Path | str = DEFAULT_REAL_DATA_ROOT,
    overwrite: bool = False,
) -> Path:
    spec = get_real_dataset_spec(dataset_name)
    output_root = Path(output_root)
    paths = dataset_raw_paths(dataset_name=spec.canonical_name, root=output_root)

    if paths.dataset_root.exists() and overwrite:
        shutil.rmtree(paths.dataset_root)
    paths.dataset_root.mkdir(parents=True, exist_ok=True)

    if spec.source_type == "openml":
        metadata = _save_openml_table(dataset_name=spec.canonical_name, output_root=output_root)
    elif spec.source_type == "uci_archive":
        metadata = _save_uci_archive(dataset_name=spec.canonical_name, output_root=output_root)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported source_type: {spec.source_type}")

    paths.metadata_path.write_text(json.dumps(jsonable_mapping(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return paths.dataset_root


def download_real_datasets(
    dataset_names: Optional[list[str]] = None,
    output_root: Path | str = DEFAULT_REAL_DATA_ROOT,
    overwrite: bool = False,
) -> Dict[str, Path]:
    names = dataset_names or list_real_dataset_names()
    return {
        name: download_real_dataset(dataset_name=name, output_root=output_root, overwrite=overwrite)
        for name in names
    }
