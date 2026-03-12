from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import as_completed
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from progress_utils import progress_bar
from parallel_utils import make_process_pool, resolve_n_jobs
from real_data import (
    create_real_data_split_manifests,
    download_real_dataset,
    load_real_binary_classification_dataset,
    prepare_real_dataset,
)
from real_data.catalog import list_real_dataset_names
from real_data.schema import (
    DEFAULT_REAL_DATA_ROOT,
    DEFAULT_REAL_PROCESSED_ROOT,
    DEFAULT_REAL_SPLIT_ROOT,
    dataset_processed_paths as classification_dataset_processed_paths,
    dataset_raw_paths as classification_dataset_raw_paths,
    dataset_split_paths as classification_dataset_split_paths,
)
from real_data.preprocess import materialize_real_dataset
from real_regression import (
    create_real_regression_split_manifests,
    download_real_regression_dataset,
    load_real_regression_dataset,
    prepare_real_regression_dataset,
)
from real_regression.catalog import list_real_regression_dataset_names
from real_regression.schema import (
    DEFAULT_REAL_REGRESSION_DATA_ROOT,
    DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
    dataset_processed_paths as regression_dataset_processed_paths,
    dataset_raw_paths as regression_dataset_raw_paths,
    dataset_split_paths as regression_dataset_split_paths,
)
from real_regression.preprocess import materialize_real_regression_dataset
from sim.grouped_classification_eval import compute_binary_classification_metrics
from sim.tabular_benchmark_models import (
    TabularBenchmarkModelConfig,
    build_tabular_benchmark_wrapper,
    expand_tabular_model_grid,
)


DEFAULT_FAMILIES = ["bagging", "rf", "gbdt", "xgb", "ctb"]
DEFAULT_SELECTION_CHECKPOINTS = [25, 50, 100, 200, 300]


def _task_label(*, task_type: str, dataset_name: str, repeat_id: int) -> str:
    return f"{str(task_type)}:{str(dataset_name)}/r{int(repeat_id):02d}"


def _progress_log(message: str) -> None:
    print(message, flush=True)


def _format_primary_metric(*, task_type: str, row: Dict[str, object], use_report_metric_for_selection: bool = False) -> str:
    metric_name = _valid_primary_metric_column(task_type, use_report_metric_for_selection=use_report_metric_for_selection)
    metric_value = row.get(metric_name)
    if metric_value is None:
        return f"{metric_name}=nan"
    return f"{metric_name}={float(metric_value):.6f}"


def ensure_open_tabular_data_ready(
    *,
    classification_datasets: Sequence[str],
    regression_datasets: Sequence[str],
    n_repeats: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    base_seed: int,
    classification_raw_root: Path,
    classification_processed_root: Path,
    classification_split_root: Path,
    regression_raw_root: Path,
    regression_processed_root: Path,
    regression_split_root: Path,
) -> None:
    for dataset_name in classification_datasets:
        _ensure_classification_dataset_ready(
            dataset_name=dataset_name,
            n_repeats=n_repeats,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            base_seed=base_seed,
            raw_root=classification_raw_root,
            processed_root=classification_processed_root,
            split_root=classification_split_root,
        )
    for dataset_name in regression_datasets:
        _ensure_regression_dataset_ready(
            dataset_name=dataset_name,
            n_repeats=n_repeats,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            base_seed=base_seed,
            raw_root=regression_raw_root,
            processed_root=regression_processed_root,
            split_root=regression_split_root,
        )


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_stored_row_count(*, manifest: Dict[str, object], full_n_rows: int) -> int:
    if bool(manifest.get("cleaned_table_is_sample", False)):
        stored = manifest.get("stored_cleaned_sample_rows")
        if stored is None:
            stored = min(5, int(full_n_rows))
        return int(stored)
    return int(full_n_rows)


def _series_is_effectively_numeric(series: pd.Series) -> bool:
    non_missing = series[pd.notna(series)]
    if non_missing.empty:
        return True
    converted = pd.to_numeric(non_missing, errors="coerce")
    return converted.notna().all()


def _normalize_string_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string")
    return normalized.fillna("__NA__").str.strip()


def _preview_frame_equals(*, stored_frame: pd.DataFrame, full_frame: pd.DataFrame, n_rows: int) -> bool:
    lhs = stored_frame.head(int(n_rows)).reset_index(drop=True).copy()
    rhs = full_frame.head(int(n_rows)).reset_index(drop=True).copy()
    if lhs.columns.astype(str).tolist() != rhs.columns.astype(str).tolist():
        return False

    for col in lhs.columns:
        lhs_col = lhs[col]
        rhs_col = rhs[col]
        if _series_is_effectively_numeric(lhs_col) and _series_is_effectively_numeric(rhs_col):
            lhs_num = pd.to_numeric(lhs_col, errors="coerce")
            rhs_num = pd.to_numeric(rhs_col, errors="coerce")
            if not lhs_num.isna().equals(rhs_num.isna()):
                return False
            if not np.allclose(lhs_num.fillna(0.0).to_numpy(dtype=float), rhs_num.fillna(0.0).to_numpy(dtype=float), rtol=0.0, atol=0.0):
                return False
            continue

        if not _normalize_string_series(lhs_col).equals(_normalize_string_series(rhs_col)):
            return False
    return True


def _validate_processed_classification_artifacts(*, dataset_name: str, raw_root: Path, processed_root: Path) -> None:
    processed_paths = classification_dataset_processed_paths(dataset_name=dataset_name, root=processed_root)
    if not processed_paths.cleaned_table_path.exists() or not processed_paths.manifest_path.exists():
        raise FileNotFoundError(f"Missing processed classification artifacts for {dataset_name!r}")

    stored_frame = pd.read_csv(processed_paths.cleaned_table_path, low_memory=False)
    stored_manifest = _read_json(processed_paths.manifest_path)
    frame, fresh_manifest = materialize_real_dataset(
        dataset_name=dataset_name, raw_root=raw_root, output_root=processed_root
    )
    if "__sample_id__" not in frame.columns or "__target__" not in frame.columns:
        raise RuntimeError(f"Processed classification table for {dataset_name!r} is missing required columns")
    if frame.empty:
        raise RuntimeError(f"Processed classification table for {dataset_name!r} is empty")
    if frame["__sample_id__"].astype(str).duplicated().any():
        raise RuntimeError(f"Processed classification table for {dataset_name!r} contains duplicate sample ids")

    y = frame["__target__"].to_numpy(dtype=int)
    labels = sorted(np.unique(y).tolist())
    if labels != [0, 1]:
        raise RuntimeError(f"Processed classification target for {dataset_name!r} must be binary 0/1; got {labels}")

    if int(stored_manifest.get("n_rows", -1)) != int(fresh_manifest.get("n_rows", -1)):
        raise RuntimeError(f"Processed manifest row count is stale for {dataset_name!r}")
    feature_columns = [str(col) for col in frame.columns if str(col) not in {"__sample_id__", "__target__"}]
    if feature_columns != list(stored_manifest.get("feature_columns", [])):
        raise RuntimeError(f"Processed manifest feature columns are stale for {dataset_name!r}")
    if list(stored_manifest.get("numeric_columns", [])) != list(fresh_manifest.get("numeric_columns", [])):
        raise RuntimeError(f"Processed manifest numeric columns are stale for {dataset_name!r}")
    if list(stored_manifest.get("categorical_columns", [])) != list(fresh_manifest.get("categorical_columns", [])):
        raise RuntimeError(f"Processed manifest categorical columns are stale for {dataset_name!r}")

    expected_rows = _expected_stored_row_count(manifest=stored_manifest, full_n_rows=int(frame.shape[0]))
    if int(stored_frame.shape[0]) != int(expected_rows):
        raise RuntimeError(f"Stored processed table preview row count is stale for {dataset_name!r}")
    if not _preview_frame_equals(stored_frame=stored_frame, full_frame=frame, n_rows=expected_rows):
        raise RuntimeError(f"Stored processed table preview is stale for {dataset_name!r}")



def _validate_processed_regression_artifacts(*, dataset_name: str, raw_root: Path, processed_root: Path) -> None:
    processed_paths = regression_dataset_processed_paths(dataset_name=dataset_name, root=processed_root)
    if not processed_paths.cleaned_table_path.exists() or not processed_paths.manifest_path.exists():
        raise FileNotFoundError(f"Missing processed regression artifacts for {dataset_name!r}")

    stored_frame = pd.read_csv(processed_paths.cleaned_table_path, low_memory=False)
    stored_manifest = _read_json(processed_paths.manifest_path)
    frame, fresh_manifest = materialize_real_regression_dataset(
        dataset_name=dataset_name, raw_root=raw_root, output_root=processed_root
    )
    if "__sample_id__" not in frame.columns or "__target__" not in frame.columns:
        raise RuntimeError(f"Processed regression table for {dataset_name!r} is missing required columns")
    if frame.empty:
        raise RuntimeError(f"Processed regression table for {dataset_name!r} is empty")
    if frame["__sample_id__"].astype(str).duplicated().any():
        raise RuntimeError(f"Processed regression table for {dataset_name!r} contains duplicate sample ids")

    y = frame["__target__"].to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise RuntimeError(f"Processed regression target for {dataset_name!r} contains non-finite values")

    if int(stored_manifest.get("n_rows", -1)) != int(fresh_manifest.get("n_rows", -1)):
        raise RuntimeError(f"Processed manifest row count is stale for {dataset_name!r}")
    feature_columns = [str(col) for col in frame.columns if str(col) not in {"__sample_id__", "__target__"}]
    if feature_columns != list(stored_manifest.get("feature_columns", [])):
        raise RuntimeError(f"Processed manifest feature columns are stale for {dataset_name!r}")
    if list(stored_manifest.get("numeric_columns", [])) != list(fresh_manifest.get("numeric_columns", [])):
        raise RuntimeError(f"Processed manifest numeric columns are stale for {dataset_name!r}")
    if list(stored_manifest.get("categorical_columns", [])) != list(fresh_manifest.get("categorical_columns", [])):
        raise RuntimeError(f"Processed manifest categorical columns are stale for {dataset_name!r}")

    expected_rows = _expected_stored_row_count(manifest=stored_manifest, full_n_rows=int(frame.shape[0]))
    if int(stored_frame.shape[0]) != int(expected_rows):
        raise RuntimeError(f"Stored processed table preview row count is stale for {dataset_name!r}")
    if not _preview_frame_equals(stored_frame=stored_frame, full_frame=frame, n_rows=expected_rows):
        raise RuntimeError(f"Stored processed table preview is stale for {dataset_name!r}")



def _validate_split_partition(*, dataset_name: str, manifest_path: Path, n_rows: int, y: Optional[np.ndarray] = None) -> None:
    manifest = _read_json(manifest_path)
    train_idx = np.asarray(manifest["train_idx"], dtype=int)
    valid_idx = np.asarray(manifest["valid_idx"], dtype=int)
    test_idx = np.asarray(manifest["test_idx"], dtype=int)

    all_idx = np.sort(np.concatenate([train_idx, valid_idx, test_idx]))
    if all_idx.shape[0] != int(n_rows) or not np.array_equal(all_idx, np.arange(int(n_rows), dtype=int)):
        raise RuntimeError(f"Split manifest {manifest_path} for {dataset_name!r} is not a partition of all rows")

    if y is not None:
        for split_name, idx in (("train", train_idx), ("valid", valid_idx), ("test", test_idx)):
            if np.unique(y[idx]).shape[0] < 2:
                raise RuntimeError(
                    f"Split manifest {manifest_path} for {dataset_name!r} yields a single-class {split_name} split"
                )
    if min(train_idx.size, valid_idx.size, test_idx.size) <= 0:
        raise RuntimeError(f"Split manifest {manifest_path} for {dataset_name!r} contains an empty split")


def _validate_classification_dataset_ready(*, dataset_name: str, n_repeats: int, raw_root: Path, processed_root: Path, split_root: Path) -> None:
    frame, _ = materialize_real_dataset(dataset_name=dataset_name, raw_root=raw_root, output_root=processed_root)
    y = frame["__target__"].to_numpy(dtype=int)
    split_paths = classification_dataset_split_paths(dataset_name=dataset_name, root=split_root)
    for repeat_id in range(int(n_repeats)):
        manifest_path = split_paths.split_dir / f"repeat_{int(repeat_id):02d}.json"
        _validate_split_partition(dataset_name=dataset_name, manifest_path=manifest_path, n_rows=frame.shape[0], y=y)
        dataset = load_real_binary_classification_dataset(
            dataset_name=dataset_name,
            repeat_id=int(repeat_id),
            group_definition="auto",
            raw_root=raw_root,
            processed_root=processed_root,
            split_root=split_root,
            random_state=int(repeat_id),
        )
        if min(dataset.train.X.shape[0], dataset.valid.X.shape[0], dataset.test.X.shape[0]) <= 0:
            raise RuntimeError(f"Loaded classification dataset {dataset_name!r} contains an empty split")



def _validate_regression_dataset_ready(*, dataset_name: str, n_repeats: int, raw_root: Path, processed_root: Path, split_root: Path) -> None:
    frame, _ = materialize_real_regression_dataset(dataset_name=dataset_name, raw_root=raw_root, output_root=processed_root)
    split_paths = regression_dataset_split_paths(dataset_name=dataset_name, root=split_root)
    for repeat_id in range(int(n_repeats)):
        manifest_path = split_paths.split_dir / f"repeat_{int(repeat_id):02d}.json"
        _validate_split_partition(dataset_name=dataset_name, manifest_path=manifest_path, n_rows=frame.shape[0], y=None)
        dataset = load_real_regression_dataset(
            dataset_name=dataset_name,
            repeat_id=int(repeat_id),
            raw_root=raw_root,
            processed_root=processed_root,
            split_root=split_root,
        )
        for split in (dataset.train, dataset.valid, dataset.test):
            if split.X.shape[0] <= 0:
                raise RuntimeError(f"Loaded regression dataset {dataset_name!r} contains an empty split")
            if not np.all(np.isfinite(split.X)) or not np.all(np.isfinite(split.y)):
                raise RuntimeError(f"Loaded regression dataset {dataset_name!r} contains non-finite values")


def run_open_tabular_benchmark(
    *,
    classification_datasets: Sequence[str],
    regression_datasets: Sequence[str],
    families: Sequence[str] = DEFAULT_FAMILIES,
    max_rounds: int = 300,
    selection_checkpoints: Sequence[int] = DEFAULT_SELECTION_CHECKPOINTS,
    max_depths: Sequence[int] = (1, 3, 5),
    min_samples_leafs: Sequence[int] = (1, 5),
    learning_rates: Sequence[float] = (0.03, 0.1),
    subsamples: Sequence[float] = (0.7, 1.0),
    colsample_bytree: Sequence[float] = (0.8,),
    ctb_inner_bootstraps: Sequence[int] = (4, 8),
    ctb_etas: Sequence[float] = (0.5, 1.0),
    ctb_instability_penalty: float = 0.0,
    ctb_weight_power: float = 1.0,
    ctb_weight_eps: float = 1e-8,
    ctb_target_modes: Sequence[str] = ("legacy",),
    ctb_curvature_eps: Sequence[float] = (1e-6,),
    n_repeats: int = 5,
    base_seed: int = 0,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    classification_raw_root: Path = DEFAULT_REAL_DATA_ROOT,
    classification_processed_root: Path = DEFAULT_REAL_PROCESSED_ROOT,
    classification_split_root: Path = DEFAULT_REAL_SPLIT_ROOT,
    regression_raw_root: Path = DEFAULT_REAL_REGRESSION_DATA_ROOT,
    regression_processed_root: Path = DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    regression_split_root: Path = DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
    output_root: Path = Path("outputs/open_tabular_benchmark"),
    n_jobs: int = 1,
    progress_log_every: int = 0,
    use_report_metric_for_selection: bool = False,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)

    selection_checkpoints = _resolved_selection_checkpoints(
        max_rounds=int(max_rounds),
        requested=selection_checkpoints,
    )
    classification_datasets = list(classification_datasets)
    regression_datasets = list(regression_datasets)
    all_runs: List[Tuple[str, str, int]] = [
        ("classification", dataset_name, repeat_id)
        for dataset_name in classification_datasets
        for repeat_id in range(int(n_repeats))
    ] + [
        ("regression", dataset_name, repeat_id)
        for dataset_name in regression_datasets
        for repeat_id in range(int(n_repeats))
    ]

    ensure_open_tabular_data_ready(
        classification_datasets=classification_datasets,
        regression_datasets=regression_datasets,
        n_repeats=n_repeats,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        base_seed=base_seed,
        classification_raw_root=Path(classification_raw_root),
        classification_processed_root=Path(classification_processed_root),
        classification_split_root=Path(classification_split_root),
        regression_raw_root=Path(regression_raw_root),
        regression_processed_root=Path(regression_processed_root),
        regression_split_root=Path(regression_split_root),
    )

    run_tasks: List[Dict[str, object]] = [
        {
            "task_type": str(task_type),
            "dataset_name": str(dataset_name),
            "repeat_id": int(repeat_id),
            "base_seed": int(base_seed),
            "families": [str(x) for x in families],
            "max_rounds": int(max_rounds),
            "selection_checkpoints": [int(x) for x in selection_checkpoints],
            "max_depths": [int(x) for x in max_depths],
            "min_samples_leafs": [int(x) for x in min_samples_leafs],
            "learning_rates": [float(x) for x in learning_rates],
            "subsamples": [float(x) for x in subsamples],
            "colsample_bytree": [float(x) for x in colsample_bytree],
            "ctb_inner_bootstraps": [int(x) for x in ctb_inner_bootstraps],
            "ctb_etas": [float(x) for x in ctb_etas],
            "ctb_instability_penalty": float(ctb_instability_penalty),
            "ctb_weight_power": float(ctb_weight_power),
            "ctb_weight_eps": float(ctb_weight_eps),
            "ctb_target_modes": [str(x) for x in ctb_target_modes],
            "ctb_curvature_eps": [float(x) for x in ctb_curvature_eps],
            "classification_raw_root": str(classification_raw_root),
            "classification_processed_root": str(classification_processed_root),
            "classification_split_root": str(classification_split_root),
            "regression_raw_root": str(regression_raw_root),
            "regression_processed_root": str(regression_processed_root),
            "regression_split_root": str(regression_split_root),
            "output_root": str(output_root),
            "progress_log_every": int(progress_log_every),
            "use_report_metric_for_selection": bool(use_report_metric_for_selection),
        }
        for task_type, dataset_name, repeat_id in all_runs
    ]

    n_jobs = resolve_n_jobs(n_jobs)
    results: List[Dict[str, object]] = []
    with progress_bar(total=len(run_tasks), desc="open-tabular benchmark", unit="run") as outer:
        if n_jobs <= 1:
            for task in run_tasks:
                results.append(_run_open_tabular_single_run(task))
                outer.update(1)
                outer.set_postfix_str(
                    f"last_done={_task_label(task_type=str(task['task_type']), dataset_name=str(task['dataset_name']), repeat_id=int(task['repeat_id']))}"
                )
        else:
            with make_process_pool(n_jobs) as executor:
                future_to_task = {executor.submit(_run_open_tabular_single_run, task): task for task in run_tasks}
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    results.append(future.result())
                    outer.update(1)
                    outer.set_postfix_str(
                        f"last_done={_task_label(task_type=str(task['task_type']), dataset_name=str(task['dataset_name']), repeat_id=int(task['repeat_id']))}"
                    )

    summary_test_rows: List[Dict[str, object]] = []
    summary_valid_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []
    for result in results:
        summary_test_rows.extend(result["summary_test_rows"])
        summary_valid_rows.extend(result["summary_valid_rows"])
        error_rows.extend(result["error_rows"])

    pd.DataFrame(summary_test_rows).to_csv(output_root / "summary_test_metrics.csv", index=False)
    pd.DataFrame(summary_valid_rows).to_csv(output_root / "summary_valid_selection.csv", index=False)
    if error_rows:
        pd.DataFrame(error_rows).to_csv(output_root / "errors.csv", index=False)

    artifact_summary = {
        "output_root": str(output_root),
        "classification_datasets": list(classification_datasets),
        "regression_datasets": list(regression_datasets),
        "families": [str(x) for x in families],
        "max_rounds": int(max_rounds),
        "selection_checkpoints": list(selection_checkpoints),
        "n_repeats": int(n_repeats),
        "base_seed": int(base_seed),
        "use_report_metric_for_selection": bool(use_report_metric_for_selection),
        "n_successful_runs": int(len(summary_test_rows)),
        "n_errors": int(len(error_rows)),
        "summary_test_metrics_path": str(output_root / "summary_test_metrics.csv"),
        "summary_valid_selection_path": str(output_root / "summary_valid_selection.csv"),
    }
    (output_root / "artifact_summary.json").write_text(
        json.dumps(artifact_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return artifact_summary


def _ensure_classification_dataset_ready(
    *,
    dataset_name: str,
    n_repeats: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    base_seed: int,
    raw_root: Path,
    processed_root: Path,
    split_root: Path,
) -> None:
    raw_paths = classification_dataset_raw_paths(dataset_name=dataset_name, root=raw_root)
    processed_paths = classification_dataset_processed_paths(dataset_name=dataset_name, root=processed_root)
    split_paths = classification_dataset_split_paths(dataset_name=dataset_name, root=split_root)

    if not raw_paths.dataset_root.exists() or not raw_paths.metadata_path.exists():
        download_real_dataset(dataset_name=dataset_name, output_root=raw_root, overwrite=False)

    needs_prepare = (
        not processed_paths.cleaned_table_path.exists()
        or not processed_paths.cleaned_table_full_path.exists()
        or not processed_paths.manifest_path.exists()
    )
    if not needs_prepare:
        try:
            _validate_processed_classification_artifacts(
                dataset_name=dataset_name, raw_root=raw_root, processed_root=processed_root
            )
        except Exception:
            needs_prepare = True
    if needs_prepare:
        prepare_real_dataset(
            dataset_name=dataset_name,
            raw_root=raw_root,
            output_root=processed_root,
            persist_full_table=True,
        )
        _validate_processed_classification_artifacts(
            dataset_name=dataset_name, raw_root=raw_root, processed_root=processed_root
        )

    manifest_paths = [split_paths.split_dir / f"repeat_{int(i):02d}.json" for i in range(int(n_repeats))]
    needs_splits = any(not path.exists() for path in manifest_paths)
    if not needs_splits:
        try:
            _validate_classification_dataset_ready(
                dataset_name=dataset_name,
                n_repeats=n_repeats,
                raw_root=raw_root,
                processed_root=processed_root,
                split_root=split_root,
            )
        except Exception:
            needs_splits = True
    if needs_splits:
        create_real_data_split_manifests(
            dataset_names=[dataset_name],
            raw_root=raw_root,
            processed_root=processed_root,
            output_root=split_root,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            n_repeats=n_repeats,
            base_seed=base_seed,
        )
    _validate_classification_dataset_ready(
        dataset_name=dataset_name,
        n_repeats=n_repeats,
        raw_root=raw_root,
        processed_root=processed_root,
        split_root=split_root,
    )


def _ensure_regression_dataset_ready(
    *,
    dataset_name: str,
    n_repeats: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    base_seed: int,
    raw_root: Path,
    processed_root: Path,
    split_root: Path,
) -> None:
    raw_paths = regression_dataset_raw_paths(dataset_name=dataset_name, root=raw_root)
    processed_paths = regression_dataset_processed_paths(dataset_name=dataset_name, root=processed_root)
    split_paths = regression_dataset_split_paths(dataset_name=dataset_name, root=split_root)

    if not raw_paths.dataset_root.exists() or not raw_paths.metadata_path.exists():
        download_real_regression_dataset(dataset_name=dataset_name, output_root=raw_root, overwrite=False)

    needs_prepare = (
        not processed_paths.cleaned_table_path.exists()
        or not processed_paths.cleaned_table_full_path.exists()
        or not processed_paths.manifest_path.exists()
    )
    if not needs_prepare:
        try:
            _validate_processed_regression_artifacts(
                dataset_name=dataset_name, raw_root=raw_root, processed_root=processed_root
            )
        except Exception:
            needs_prepare = True
    if needs_prepare:
        prepare_real_regression_dataset(
            dataset_name=dataset_name,
            raw_root=raw_root,
            output_root=processed_root,
            persist_full_table=True,
        )
        _validate_processed_regression_artifacts(
            dataset_name=dataset_name, raw_root=raw_root, processed_root=processed_root
        )

    manifest_paths = [split_paths.split_dir / f"repeat_{int(i):02d}.json" for i in range(int(n_repeats))]
    needs_splits = any(not path.exists() for path in manifest_paths)
    if not needs_splits:
        try:
            _validate_regression_dataset_ready(
                dataset_name=dataset_name,
                n_repeats=n_repeats,
                raw_root=raw_root,
                processed_root=processed_root,
                split_root=split_root,
            )
        except Exception:
            needs_splits = True
    if needs_splits:
        create_real_regression_split_manifests(
            dataset_names=[dataset_name],
            raw_root=raw_root,
            processed_root=processed_root,
            output_root=split_root,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            n_repeats=n_repeats,
            base_seed=base_seed,
        )
    _validate_regression_dataset_ready(
        dataset_name=dataset_name,
        n_repeats=n_repeats,
        raw_root=raw_root,
        processed_root=processed_root,
        split_root=split_root,
    )


def _load_task_dataset(
    *,
    task_type: str,
    dataset_name: str,
    repeat_id: int,
    run_seed: int,
    classification_raw_root: Path,
    classification_processed_root: Path,
    classification_split_root: Path,
    regression_raw_root: Path,
    regression_processed_root: Path,
    regression_split_root: Path,
):
    if str(task_type).strip().lower() == "classification":
        return load_real_binary_classification_dataset(
            dataset_name=dataset_name,
            repeat_id=int(repeat_id),
            group_definition="auto",
            raw_root=classification_raw_root,
            processed_root=classification_processed_root,
            split_root=classification_split_root,
            random_state=int(run_seed),
        )
    if str(task_type).strip().lower() == "regression":
        return load_real_regression_dataset(
            dataset_name=dataset_name,
            repeat_id=int(repeat_id),
            raw_root=regression_raw_root,
            processed_root=regression_processed_root,
            split_root=regression_split_root,
        )
    raise ValueError(f"Unsupported task_type={task_type!r}")


def _run_open_tabular_single_run(task: Dict[str, object]) -> Dict[str, object]:
    task_type = str(task["task_type"])
    dataset_name = str(task["dataset_name"])
    repeat_id = int(task["repeat_id"])
    run_seed = int(task["base_seed"]) + int(repeat_id)

    summary_test_rows: List[Dict[str, object]] = []
    summary_valid_rows: List[Dict[str, object]] = []
    error_rows: List[Dict[str, object]] = []

    try:
        dataset = _load_task_dataset(
            task_type=task_type,
            dataset_name=dataset_name,
            repeat_id=repeat_id,
            run_seed=run_seed,
            classification_raw_root=Path(task["classification_raw_root"]),
            classification_processed_root=Path(task["classification_processed_root"]),
            classification_split_root=Path(task["classification_split_root"]),
            regression_raw_root=Path(task["regression_raw_root"]),
            regression_processed_root=Path(task["regression_processed_root"]),
            regression_split_root=Path(task["regression_split_root"]),
        )
    except Exception as exc:
        error_rows.append(
            {
                "task_type": task_type,
                "dataset_name": dataset_name,
                "repeat_id": int(repeat_id),
                "family": "__dataset_load__",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        return {
            "summary_test_rows": summary_test_rows,
            "summary_valid_rows": summary_valid_rows,
            "error_rows": error_rows,
        }

    dataset_output_dir = Path(task["output_root"]) / dataset_name / f"repeat_{int(repeat_id):02d}"
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    progress_log_every = int(task.get("progress_log_every", 0))
    if progress_log_every > 0:
        _progress_log(
            f"[run:start] {_task_label(task_type=task_type, dataset_name=dataset_name, repeat_id=int(repeat_id))} "
            f"seed={int(run_seed)} families={','.join(str(x) for x in task['families'])}"
        )

    for family in task["families"]:
        try:
            family_configs = expand_tabular_model_grid(
                task_type=task_type,
                families=[str(family)],
                max_depths=task["max_depths"],
                n_estimators=int(task["max_rounds"]),
                min_samples_leafs=task["min_samples_leafs"],
                learning_rates=task["learning_rates"],
                subsamples=task["subsamples"],
                colsample_bytree=task["colsample_bytree"],
                inner_bootstraps=task["ctb_inner_bootstraps"],
                etas=task["ctb_etas"],
                instability_penalty=float(task["ctb_instability_penalty"]),
                weight_power=float(task["ctb_weight_power"]),
                weight_eps=float(task["ctb_weight_eps"]),
                ctb_target_modes=task["ctb_target_modes"],
                ctb_curvature_eps=task["ctb_curvature_eps"],
                random_state=run_seed,
            )
            result = _run_family_grid_search(
                task_type=task_type,
                dataset_name=dataset_name,
                repeat_id=repeat_id,
                run_seed=run_seed,
                dataset=dataset,
                family=str(family),
                configs=family_configs,
                selection_checkpoints=task["selection_checkpoints"],
                output_dir=dataset_output_dir / str(family),
                show_progress=False,
                progress_log_every=progress_log_every,
                use_report_metric_for_selection=bool(task.get("use_report_metric_for_selection", False)),
            )
            summary_test_rows.append(result["test_summary_row"])
            summary_valid_rows.append(result["valid_summary_row"])
        except Exception as exc:
            error_rows.append(
                {
                    "task_type": task_type,
                    "dataset_name": dataset_name,
                    "repeat_id": int(repeat_id),
                    "family": str(family),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    if progress_log_every > 0:
        _progress_log(
            f"[run:done] {_task_label(task_type=task_type, dataset_name=dataset_name, repeat_id=int(repeat_id))} "
            f"families_done={len(summary_test_rows)}/{len(task['families'])} errors={len(error_rows)}"
        )

    return {
        "summary_test_rows": summary_test_rows,
        "summary_valid_rows": summary_valid_rows,
        "error_rows": error_rows,
    }


def _run_family_grid_search(
    *,
    task_type: str,
    dataset_name: str,
    repeat_id: int,
    run_seed: int,
    dataset,
    family: str,
    configs: Sequence[TabularBenchmarkModelConfig],
    selection_checkpoints: Sequence[int],
    output_dir: Path,
    show_progress: bool = True,
    progress_log_every: int = 0,
    use_report_metric_for_selection: bool = False,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    best_row: Optional[Dict[str, object]] = None
    best_config: Optional[TabularBenchmarkModelConfig] = None
    best_wrapper = None
    best_test_pred: Optional[np.ndarray] = None

    desc = f"{dataset_name}/r{int(repeat_id):02d}/{family}"
    if show_progress:
        bar_context = progress_bar(total=len(configs), desc=desc, unit="cfg", leave=False)
    else:
        bar_context = None

    if progress_log_every > 0:
        _progress_log(
            f"[family:start] {_task_label(task_type=task_type, dataset_name=dataset_name, repeat_id=int(repeat_id))} "
            f"family={str(family)} n_cfg={len(configs)}"
        )

    with bar_context if bar_context is not None else nullcontext() as bar:
        for config_idx, config in enumerate(configs, start=1):
            wrapper = build_tabular_benchmark_wrapper(
                config=config,
                selection_checkpoints=selection_checkpoints,
                use_report_metric_for_selection=use_report_metric_for_selection,
            )
            wrapper.fit(dataset.train, dataset.valid)
            valid_pred = _predict_for_task(task_type=task_type, wrapper=wrapper, split=dataset.valid)
            test_pred = _predict_for_task(task_type=task_type, wrapper=wrapper, split=dataset.test)
            valid_metrics = _compute_task_metrics(task_type=task_type, y_true=dataset.valid.y, prediction=valid_pred)
            test_metrics = _compute_task_metrics(task_type=task_type, y_true=dataset.test.y, prediction=test_pred)
            row = {
                "task_type": str(task_type),
                "dataset_name": str(dataset_name),
                "repeat_id": int(repeat_id),
                "seed": int(run_seed),
                "family": str(family),
                "model_name": config.model_name,
                "selected_checkpoint": int(wrapper.selected_checkpoint_),
                **config.to_dict(),
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            rows.append(row)
            if best_row is None or _selection_key(
                task_type=task_type,
                row=row,
                use_report_metric_for_selection=use_report_metric_for_selection,
            ) < _selection_key(
                task_type=task_type,
                row=best_row,
                use_report_metric_for_selection=use_report_metric_for_selection,
            ):
                best_row = row
                best_config = config
                best_wrapper = wrapper
                best_test_pred = np.asarray(test_pred)
            if bar is not None:
                bar.update(1)
            if progress_log_every > 0 and (config_idx == 1 or config_idx % progress_log_every == 0 or config_idx == len(configs)):
                _progress_log(
                    f"[family:progress] {_task_label(task_type=task_type, dataset_name=dataset_name, repeat_id=int(repeat_id))} "
                    f"family={str(family)} cfg={config_idx}/{len(configs)} current={config.model_name} "
                    f"best={best_config.model_name if best_config is not None else 'n/a'} "
                    f"{_format_primary_metric(task_type=task_type, row=best_row if best_row is not None else row, use_report_metric_for_selection=use_report_metric_for_selection)}"
                )

    if best_row is None or best_config is None or best_wrapper is None or best_test_pred is None:
        raise RuntimeError(f"No successful model fits for dataset={dataset_name!r}, repeat_id={repeat_id}, family={family!r}")

    sort_metric = _valid_primary_metric_column(
        task_type,
        use_report_metric_for_selection=use_report_metric_for_selection,
    )
    pd.DataFrame(rows).sort_values(
        by=[sort_metric, "selected_checkpoint", "max_depth", "min_samples_leaf"],
        ascending=[
            not _selection_metric_higher_is_better(
                task_type,
                use_report_metric_for_selection=use_report_metric_for_selection,
            ),
            True,
            True,
            False,
        ],
    ).reset_index(drop=True).to_csv(output_dir / "grid_search_results.csv", index=False)
    best_wrapper.selection_trace_.to_csv(output_dir / "valid_selection_trace.csv", index=False)

    best_payload = {
        **best_config.to_dict(),
        "task_type": str(task_type),
        "dataset_name": str(dataset_name),
        "repeat_id": int(repeat_id),
        "seed": int(run_seed),
        "family": str(family),
        "selected_checkpoint": int(best_wrapper.selected_checkpoint_),
        "selection_metric": _valid_primary_metric_column(
            task_type,
            use_report_metric_for_selection=use_report_metric_for_selection,
        ),
        "valid_metrics": {k.replace("valid_", ""): v for k, v in best_row.items() if str(k).startswith("valid_")},
        "test_metrics": {k.replace("test_", ""): v for k, v in best_row.items() if str(k).startswith("test_")},
    }
    (output_dir / "best_config.json").write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(
        json.dumps(best_payload["test_metrics"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    valid_summary_row = {
        "task_type": str(task_type),
        "dataset_name": str(dataset_name),
        "repeat_id": int(repeat_id),
        "seed": int(run_seed),
        "family": str(family),
        "model_name": best_config.model_name,
        "selected_checkpoint": int(best_wrapper.selected_checkpoint_),
        **best_config.to_dict(),
        **{k: v for k, v in best_row.items() if str(k).startswith("valid_")},
    }
    test_summary_row = {
        **valid_summary_row,
        **{k: v for k, v in best_row.items() if str(k).startswith("test_")},
    }
    if progress_log_every > 0:
        _progress_log(
            f"[family:done] {_task_label(task_type=task_type, dataset_name=dataset_name, repeat_id=int(repeat_id))} "
            f"family={str(family)} best={best_config.model_name} checkpoint={int(best_wrapper.selected_checkpoint_)} "
            f"{_format_primary_metric(task_type=task_type, row=best_row, use_report_metric_for_selection=use_report_metric_for_selection)}"
        )
    return {"valid_summary_row": valid_summary_row, "test_summary_row": test_summary_row}


def _predict_for_task(*, task_type: str, wrapper, split) -> np.ndarray:
    if str(task_type).strip().lower() == "classification":
        return np.asarray(wrapper.predict_proba(split.X), dtype=float)
    return np.asarray(wrapper.predict(split.X), dtype=float)


def _compute_task_metrics(*, task_type: str, y_true, prediction) -> Dict[str, float]:
    if str(task_type).strip().lower() == "classification":
        metrics = compute_binary_classification_metrics(y_true=y_true, y_prob=prediction)
        return {str(k): float(v) for k, v in metrics.items()}
    y_true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(prediction, dtype=float)
    mse = float(mean_squared_error(y_true_arr, pred_arr))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true_arr, pred_arr)),
        "r2": float(r2_score(y_true_arr, pred_arr)),
    }

def _make_prediction_frame(
    *,
    task_type: str,
    dataset_name: str,
    repeat_id: int,
    family: str,
    split_name: str,
    selected_checkpoint: int,
    sample_id,
    y_true,
    prediction,
) -> pd.DataFrame:
    sample_id_arr = np.asarray(sample_id, dtype=object)
    y_true_arr = np.asarray(y_true)
    pred_arr = np.asarray(prediction)
    frame = pd.DataFrame(
        {
            "dataset_name": str(dataset_name),
            "repeat_id": int(repeat_id),
            "family": str(family),
            "split": str(split_name),
            "selected_checkpoint": int(selected_checkpoint),
            "sample_id": sample_id_arr,
            "y_true": y_true_arr,
        }
    )
    if str(task_type).strip().lower() == "classification":
        prob = np.clip(pred_arr.astype(float), 1e-8, 1.0 - 1e-8)
        frame["y_prob"] = prob
        frame["y_pred"] = (prob >= 0.5).astype(int)
        frame["log_loss_i"] = -(y_true_arr.astype(int) * np.log(prob) + (1 - y_true_arr.astype(int)) * np.log(1.0 - prob))
        frame["brier_i"] = (prob - y_true_arr.astype(int)) ** 2
        return frame
    pred = pred_arr.astype(float)
    frame["y_pred"] = pred
    frame["residual"] = y_true_arr.astype(float) - pred
    frame["abs_error"] = np.abs(frame["residual"].to_numpy(dtype=float))
    return frame

def _resolved_selection_checkpoints(*, max_rounds: int, requested: Sequence[int]) -> List[int]:
    checkpoints = {int(x) for x in requested if 0 < int(x) <= int(max_rounds)}
    checkpoints.add(int(max_rounds))
    return sorted(checkpoints)


def _valid_primary_metric_column(task_type: str, use_report_metric_for_selection: bool = False) -> str:
    normalized_task = str(task_type).strip().lower()
    if normalized_task == "classification":
        return "valid_accuracy" if use_report_metric_for_selection else "valid_log_loss"
    return "valid_mse" if use_report_metric_for_selection else "valid_rmse"


def _selection_metric_higher_is_better(task_type: str, use_report_metric_for_selection: bool = False) -> bool:
    return str(task_type).strip().lower() == "classification" and bool(use_report_metric_for_selection)


def _selection_key(
    *,
    task_type: str,
    row: Dict[str, object],
    use_report_metric_for_selection: bool = False,
) -> Tuple[float, int, int, int]:
    metric_name = _valid_primary_metric_column(task_type, use_report_metric_for_selection=use_report_metric_for_selection)
    metric_value = float(row[metric_name])
    if _selection_metric_higher_is_better(task_type, use_report_metric_for_selection=use_report_metric_for_selection):
        metric_value = -metric_value
    return (
        metric_value,
        int(row["selected_checkpoint"]),
        int(row["max_depth"]),
        -int(row["min_samples_leaf"]),
    )


def default_classification_datasets() -> List[str]:
    return list_real_dataset_names()


def default_regression_datasets() -> List[str]:
    return list_real_regression_dataset_names()