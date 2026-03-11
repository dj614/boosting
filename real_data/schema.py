from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RealDatasetSpec:
    canonical_name: str
    task_type: str
    source_type: str
    target_column: str
    positive_aliases: Tuple[str, ...] = field(default_factory=tuple)
    positive_label: str
    openml_name: Optional[str] = None
    openml_version: Optional[int] = None
    uci_download_url: Optional[str] = None
    archive_member: Optional[str] = None
    default_group_rules: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


@dataclass(frozen=True)
class RawDatasetPaths:
    dataset_root: Path
    metadata_path: Path
    raw_table_path: Optional[Path]
    raw_archive_path: Optional[Path]
    extracted_dir: Path

@dataclass(frozen=True)
class ProcessedDatasetPaths:
    dataset_root: Path
    cleaned_table_path: Path
    manifest_path: Path

DEFAULT_REAL_DATA_ROOT = Path("data") / "raw" / "real"
DEFAULT_REAL_PROCESSED_ROOT = Path("data") / "processed" / "real"


def dataset_raw_paths(dataset_name: str, root: Path | str = DEFAULT_REAL_DATA_ROOT) -> RawDatasetPaths:
    base_root = Path(root)
    dataset_root = base_root / dataset_name
    return RawDatasetPaths(
        dataset_root=dataset_root,
        metadata_path=dataset_root / "metadata.json",
        raw_table_path=dataset_root / "raw_table.csv",
        raw_archive_path=dataset_root / "source_archive",
        extracted_dir=dataset_root / "extracted",
    )


def dataset_processed_paths(
    dataset_name: str,
    root: Path | str = DEFAULT_REAL_PROCESSED_ROOT,
) -> ProcessedDatasetPaths:
    base_root = Path(root)
    dataset_root = base_root / dataset_name
    return ProcessedDatasetPaths(
        dataset_root=dataset_root,
        cleaned_table_path=dataset_root / "cleaned_table.csv",
        manifest_path=dataset_root / "manifest.json",
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
