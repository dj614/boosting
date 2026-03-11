from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

@dataclass(frozen=True)
class RealRegressionDatasetSpec:
    canonical_name: str
    task_type: str
    source_type: str
    target_column: str
    sklearn_dataset_name: Optional[str] = None
    uci_dataset_id: Optional[int] = None
    folktables_dataset_name: Optional[str] = None
    folktables_year: Optional[int] = None
    feature_columns: Tuple[str, ...] = field(default_factory=tuple)
    categorical_columns: Tuple[str, ...] = field(default_factory=tuple)
    default_split_strategy: str = "quantile_stratified"
    notes: str = ""

@dataclass(frozen=True)
class RawRegressionDatasetPaths:
    dataset_root: Path
    metadata_path: Path
    raw_table_path: Optional[Path]
    raw_archive_path: Optional[Path]
    extracted_dir: Path

@dataclass(frozen=True)
class ProcessedRegressionDatasetPaths:
    dataset_root: Path
    cleaned_table_path: Path
    manifest_path: Path

@dataclass(frozen=True)
class RegressionSplitManifestPaths:
    dataset_root: Path
    split_dir: Path

DEFAULT_REAL_REGRESSION_DATA_ROOT = Path("data") / "raw" / "real_regression"
DEFAULT_REAL_REGRESSION_PROCESSED_ROOT = Path("data") / "processed" / "real_regression"
DEFAULT_REAL_REGRESSION_SPLIT_ROOT = Path("data") / "processed" / "real_regression" / "split_manifests"

def dataset_raw_paths(
    dataset_name: str,
    root: Path | str = DEFAULT_REAL_REGRESSION_DATA_ROOT,
) -> RawRegressionDatasetPaths:
    base_root = Path(root)
    dataset_root = base_root / dataset_name
    return RawRegressionDatasetPaths(
        dataset_root=dataset_root,
        metadata_path=dataset_root / "metadata.json",
        raw_table_path=dataset_root / "raw_table.csv",
        raw_archive_path=dataset_root / "source_archive",
        extracted_dir=dataset_root / "extracted",
    )

def dataset_processed_paths(
    dataset_name: str,
    root: Path | str = DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
) -> ProcessedRegressionDatasetPaths:
    base_root = Path(root)
    dataset_root = base_root / dataset_name
    return ProcessedRegressionDatasetPaths(
        dataset_root=dataset_root,
        cleaned_table_path=dataset_root / "cleaned_table.csv",
        manifest_path=dataset_root / "manifest.json",
    )

def dataset_split_paths(
    dataset_name: str,
    root: Path | str = DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
) -> RegressionSplitManifestPaths:
    base_root = Path(root)
    dataset_root = base_root / dataset_name
    return RegressionSplitManifestPaths(
        dataset_root=dataset_root,
        split_dir=dataset_root,
    )

def ensure_parent_dirs(paths: Sequence[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)

def jsonable_mapping(mapping: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in mapping.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out
