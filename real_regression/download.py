from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlopen

import pandas as pd

try:  # pragma: no cover - sklearn import is environment-dependent
    from sklearn.datasets import fetch_california_housing
except Exception:  # pragma: no cover
    fetch_california_housing = None

try:  # pragma: no cover - optional dependency
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

from .catalog import get_real_regression_dataset_spec, list_real_regression_dataset_names
from .schema import DEFAULT_REAL_REGRESSION_DATA_ROOT, dataset_raw_paths, ensure_parent_dirs, jsonable_mapping


_UCI_DOWNLOAD_URLS = {
    165: "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
    464: "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip",
}

_UCI_ARCHIVE_MEMBERS = {
    165: "Concrete_Data.xls",
    464: "train.csv",
}

_ACS_STATE_LIST = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR",
]
_ACS_STATE_CODES = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56", "PR": "72",
}
_ACS_CHUNKSIZE = 200_000

_RAW_SAMPLE_ROWS = 5
_RAW_FRAME_CACHE: Dict[tuple[str, str], pd.DataFrame] = {}


def _cache_key(dataset_name: str, output_root: Path) -> tuple[str, str]:
    return str(dataset_name), str(Path(output_root).resolve())


def get_cached_raw_frame(dataset_name: str, output_root: Path | str = DEFAULT_REAL_REGRESSION_DATA_ROOT) -> Optional[pd.DataFrame]:
    frame = _RAW_FRAME_CACHE.get(_cache_key(dataset_name=dataset_name, output_root=Path(output_root)))
    if frame is None:
        return None
    return frame.copy()

def _download_bytes(url: str) -> bytes:
    with urlopen(url) as response:  # nosec - downloading known public datasets
        return response.read()

def _write_raw_table_sample(dataset_name: str, output_root: Path, frame: pd.DataFrame) -> Path:
    paths = dataset_raw_paths(dataset_name=dataset_name, root=output_root)
    ensure_parent_dirs([paths.raw_table_path])
    frame.head(_RAW_SAMPLE_ROWS).to_csv(paths.raw_table_path, index=False)
    _RAW_FRAME_CACHE[_cache_key(dataset_name=dataset_name, output_root=output_root)] = frame
    return paths.raw_table_path


def _acs_state_file_name(*, year: int, state: str, survey: str = "person") -> str:
    survey_code = "p" if str(survey).strip().lower() == "person" else "h"
    state_key = str(state).strip().upper()
    if int(year) >= 2017:
        return f"psam_{survey_code}{_ACS_STATE_CODES[state_key]}.csv"
    return f"ss{str(year)[-2:]}{survey_code}{state_key.lower()}.csv"


def _ensure_acs_state_file(*, year: int, horizon: str, survey: str, state: str, root_dir: Path) -> Path:
    state_key = str(state).strip().upper()
    if state_key not in _ACS_STATE_CODES:
        raise ValueError(f"Unsupported ACS state code: {state!r}")
    if str(horizon) not in {"1-Year", "5-Year"}:
        raise ValueError(f"Unsupported ACS horizon: {horizon!r}")
    if str(survey).strip().lower() not in {"person", "household"}:
        raise ValueError(f"Unsupported ACS survey type: {survey!r}")

    datadir = Path(root_dir) / str(year) / str(horizon)
    datadir.mkdir(parents=True, exist_ok=True)
    file_name = _acs_state_file_name(year=int(year), state=state_key, survey=survey)
    file_path = datadir / file_name
    if file_path.exists():
        return file_path

    survey_code = "p" if str(survey).strip().lower() == "person" else "h"
    remote_name = f"csv_{survey_code}{state_key.lower()}.zip"
    base_url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{int(year)}/{horizon}"
    payload = _download_bytes(f"{base_url}/{remote_name}")
    with zipfile.ZipFile(io.BytesIO(payload), mode="r") as zf:
        zf.extract(file_name, path=datadir)
    return file_path


def _load_selected_acs_columns(*, year: int, horizon: str, survey: str, states: list[str], required_columns: list[str], root_dir: Path) -> pd.DataFrame:
    dtype_map = {str(col): "float32" for col in required_columns if str(col) != "PINCP"}
    if "PINCP" in required_columns:
        dtype_map["PINCP"] = "float64"

    frames: list[pd.DataFrame] = []
    for state in states:
        file_path = _ensure_acs_state_file(year=int(year), horizon=horizon, survey=survey, state=state, root_dir=root_dir)
        chunk_iter = pd.read_csv(
            file_path,
            usecols=required_columns,
            dtype=dtype_map,
            chunksize=_ACS_CHUNKSIZE,
            low_memory=True,
        )
        for chunk in chunk_iter:
            frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=required_columns)
    return pd.concat(frames, ignore_index=True)


def _save_sklearn_table(dataset_name: str, output_root: Path) -> Dict[str, object]:
    if fetch_california_housing is None:  # pragma: no cover
        raise ImportError("scikit-learn fetch_california_housing is unavailable in this environment")

    spec = get_real_regression_dataset_spec(dataset_name)
    if spec.sklearn_dataset_name != "fetch_california_housing":
        raise ValueError(f"Unsupported sklearn regression dataset: {spec.sklearn_dataset_name!r}")

    bunch = fetch_california_housing(as_frame=True)
    frame = bunch.frame.copy()
    paths = dataset_raw_paths(dataset_name=dataset_name, root=output_root)
    ensure_parent_dirs([paths.raw_table_path, paths.metadata_path])
    sample_path = _write_raw_table_sample(dataset_name=dataset_name, output_root=output_root, frame=frame)

    metadata = {
        "dataset_name": spec.canonical_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "target_column": spec.target_column,
        "sklearn_dataset_name": spec.sklearn_dataset_name,
        "raw_table_path": str(sample_path),
        "raw_table_is_sample": True,
        "raw_table_sample_rows": int(min(_RAW_SAMPLE_ROWS, frame.shape[0])),
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "raw_columns": frame.columns.astype(str).tolist(),
        "notes": spec.notes,
    }
    return metadata


def _save_folktables_table(dataset_name: str, output_root: Path) -> Dict[str, object]:
    spec = get_real_regression_dataset_spec(dataset_name)
    if spec.folktables_year is None:
        raise ValueError(f"Dataset {dataset_name!r} does not define folktables_year")
    if not spec.feature_columns:
        raise ValueError(f"Dataset {dataset_name!r} does not define feature_columns")

    required_columns = list(spec.feature_columns) + [spec.target_column]
    frame = _load_selected_acs_columns(
        year=int(spec.folktables_year),
        horizon="1-Year",
        survey="person",
        states=list(_ACS_STATE_LIST),
        required_columns=required_columns,
        root_dir=Path(output_root),
    )
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise KeyError(f"ACS source data for {dataset_name!r} is missing columns: {missing}")

    paths = dataset_raw_paths(dataset_name=dataset_name, root=output_root)
    ensure_parent_dirs([paths.raw_table_path, paths.metadata_path])
    sample_path = _write_raw_table_sample(dataset_name=dataset_name, output_root=output_root, frame=frame)

    metadata = {
        "dataset_name": spec.canonical_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "target_column": spec.target_column,
        "folktables_dataset_name": spec.folktables_dataset_name,
        "folktables_year": spec.folktables_year,
        "feature_columns": list(spec.feature_columns),
        "categorical_columns": list(spec.categorical_columns),
        "raw_table_path": str(sample_path),
        "raw_table_is_sample": True,
        "raw_table_sample_rows": int(min(_RAW_SAMPLE_ROWS, frame.shape[0])),
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "raw_columns": frame.columns.astype(str).tolist(),
        "notes": spec.notes,
    }
    return metadata


def _save_uci_archive(dataset_name: str, output_root: Path) -> Dict[str, object]:
    spec = get_real_regression_dataset_spec(dataset_name)
    if spec.uci_dataset_id is None:
        raise ValueError(f"Dataset {dataset_name!r} does not define a UCI dataset id")

    download_url = _UCI_DOWNLOAD_URLS.get(int(spec.uci_dataset_id))
    archive_member = _UCI_ARCHIVE_MEMBERS.get(int(spec.uci_dataset_id))
    if not download_url or not archive_member:
        raise ValueError(f"Unsupported UCI regression dataset id: {spec.uci_dataset_id}")

    paths = dataset_raw_paths(dataset_name=dataset_name, root=output_root)
    archive_suffix = Path(download_url).suffix or ".zip"
    archive_path = paths.raw_archive_path.with_suffix(archive_suffix)
    ensure_parent_dirs([archive_path, paths.metadata_path])
    paths.extracted_dir.mkdir(parents=True, exist_ok=True)

    payload = _download_bytes(download_url)
    archive_path.write_bytes(payload)

    extracted_members = []
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, mode="r") as zf:
            zf.extractall(paths.extracted_dir)
            extracted_members = sorted(zf.namelist())
    else:  # pragma: no cover
        raise ValueError(f"Unsupported archive suffix for {dataset_name!r}: {archive_path.suffix}")

    member_path = paths.extracted_dir / archive_member
    if not member_path.exists():
        raise FileNotFoundError(
            f"Expected archive member for {dataset_name!r} not found: {member_path}. "
            f"Extracted members were: {extracted_members}"
        )

    if member_path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(member_path)
    elif member_path.suffix.lower() == ".csv":
        frame = pd.read_csv(member_path, low_memory=False)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported archive member suffix for {dataset_name!r}: {member_path.suffix}")

    sample_path = _write_raw_table_sample(dataset_name=dataset_name, output_root=output_root, frame=frame)

    metadata = {
        "dataset_name": spec.canonical_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "target_column": spec.target_column,
        "uci_dataset_id": spec.uci_dataset_id,
        "raw_archive_path": str(archive_path),
        "archive_member": archive_member,
        "raw_table_path": str(sample_path),
        "raw_table_is_sample": True,
        "raw_table_sample_rows": int(min(_RAW_SAMPLE_ROWS, frame.shape[0])),
        "extracted_dir": str(paths.extracted_dir),
        "extracted_members": extracted_members,
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "raw_columns": frame.columns.astype(str).tolist(),
        "notes": spec.notes,
    }
    return metadata


def _save_manual_dataset(dataset_name: str, output_root: Path) -> Dict[str, object]:
    spec = get_real_regression_dataset_spec(dataset_name)
    paths = dataset_raw_paths(dataset_name=dataset_name, root=output_root)
    ensure_parent_dirs([paths.raw_table_path, paths.metadata_path])

    if dataset_name == "diamonds":
        if sns is None:  # pragma: no cover
            raise ImportError("seaborn is required to download the diamonds dataset")
        frame = sns.load_dataset("diamonds")
    else:  # pragma: no cover
        raise ValueError(f"Unsupported manual regression dataset: {dataset_name!r}")

    sample_path = _write_raw_table_sample(dataset_name=dataset_name, output_root=output_root, frame=frame)
    metadata = {
        "dataset_name": spec.canonical_name,
        "task_type": spec.task_type,
        "source_type": spec.source_type,
        "target_column": spec.target_column,
        "raw_table_path": str(sample_path),
        "raw_table_is_sample": True,
        "raw_table_sample_rows": int(min(_RAW_SAMPLE_ROWS, frame.shape[0])),
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "raw_columns": frame.columns.astype(str).tolist(),
        "notes": spec.notes,
    }
    return metadata


def download_real_regression_dataset(
    dataset_name: str,
    output_root: Path | str = DEFAULT_REAL_REGRESSION_DATA_ROOT,
    overwrite: bool = False,
) -> Path:
    spec = get_real_regression_dataset_spec(dataset_name)
    output_root = Path(output_root)
    paths = dataset_raw_paths(dataset_name=spec.canonical_name, root=output_root)

    if paths.dataset_root.exists() and overwrite:
        shutil.rmtree(paths.dataset_root)
    paths.dataset_root.mkdir(parents=True, exist_ok=True)

    if spec.source_type == "sklearn":
        metadata = _save_sklearn_table(dataset_name=spec.canonical_name, output_root=output_root)
    elif spec.source_type == "folktables":
        metadata = _save_folktables_table(dataset_name=spec.canonical_name, output_root=output_root)
    elif spec.source_type == "uci":
        metadata = _save_uci_archive(dataset_name=spec.canonical_name, output_root=output_root)
    elif spec.source_type == "manual_download":
        metadata = _save_manual_dataset(dataset_name=spec.canonical_name, output_root=output_root)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported source_type: {spec.source_type}")

    paths.metadata_path.write_text(json.dumps(jsonable_mapping(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return paths.dataset_root


def download_real_regression_datasets(
    dataset_names: Optional[list[str]] = None,
    output_root: Path | str = DEFAULT_REAL_REGRESSION_DATA_ROOT,
    overwrite: bool = False,
) -> Dict[str, Path]:
    names = dataset_names or list_real_regression_dataset_names()
    return {
        name: download_real_regression_dataset(dataset_name=name, output_root=output_root, overwrite=overwrite)
        for name in names
    }
