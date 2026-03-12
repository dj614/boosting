from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .catalog import get_real_dataset_spec
from .schema import DEFAULT_REAL_PROCESSED_ROOT, DEFAULT_REAL_SPLIT_ROOT, dataset_processed_paths, dataset_split_paths
from sim.grouped_classification_data import (
    BinaryClassificationDataset,
    _build_preprocessor,
    _feature_names_from_preprocessor,
    _make_split,
    with_margin_based_difficulty_groups,
)


Array = np.ndarray


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_numpy_strings(values) -> Array:
    return np.asarray([str(v) for v in values], dtype=object)


def _slice_metadata(metadata: Mapping[str, Array], index: Array) -> Dict[str, Array]:
    out: Dict[str, Array] = {}
    for key, value in metadata.items():
        out[key] = np.asarray(value)[index]
    return out


def _coerce_categorical_columns(frame: pd.DataFrame, categorical_columns) -> pd.DataFrame:
    categorical_set = {str(col) for col in categorical_columns if str(col) in frame.columns}
    if not categorical_set:
        return frame
    out = frame.copy()
    for col in out.columns:
        if str(col) in categorical_set:
            out[col] = out[col].astype("string")
    return out


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


def _resolve_group_definition(dataset_name: str, group_definition: str) -> str:
    spec = get_real_dataset_spec(dataset_name)
    requested = str(group_definition).strip().lower()
    if requested != "auto":
        return requested

    defaults = [str(x).strip().lower() for x in spec.default_group_rules]
    for candidate in defaults:
        if candidate != "difficulty_group":
            return candidate
    return "difficulty_group"


def _group_from_rule(frame: pd.DataFrame, dataset_name: str, group_definition: str) -> pd.Series:
    rule = _resolve_group_definition(dataset_name=dataset_name, group_definition=group_definition)

    if rule == "difficulty_group":
        return pd.Series(["all_samples"] * frame.shape[0], index=frame.index, dtype="object")

    if rule == "sex":
        if "sex" not in frame.columns:
            raise ValueError(f"Dataset {dataset_name!r} does not contain a 'sex' column")
        return frame["sex"].fillna("missing").astype(str)

    if rule == "pclass":
        if "pclass" not in frame.columns:
            raise ValueError(f"Dataset {dataset_name!r} does not contain a 'pclass' column")
        return frame["pclass"].fillna("missing").astype(str).map(lambda x: f"pclass_{x}")

    if rule == "sex_pclass":
        sex = _group_from_rule(frame=frame, dataset_name=dataset_name, group_definition="sex")
        pclass = _group_from_rule(frame=frame, dataset_name=dataset_name, group_definition="pclass")
        return sex.astype(str) + "__" + pclass.astype(str)

    raise ValueError(
        f"Unsupported group_definition={group_definition!r} for dataset {dataset_name!r}. "
        "Supported values are: auto, difficulty_group, sex, pclass, sex_pclass."
    )


def _assemble_from_fixed_splits(
    *,
    dataset_name: str,
    raw_features: pd.DataFrame,
    y: Array,
    group: Array,
    sample_id: Array,
    train_idx: Array,
    valid_idx: Array,
    test_idx: Array,
    metadata_frame: Optional[pd.DataFrame],
) -> BinaryClassificationDataset:
    y_arr = np.asarray(y, dtype=int)
    group_arr = _as_numpy_strings(group)
    sample_id_arr = _as_numpy_strings(sample_id)
    train_idx = np.asarray(train_idx, dtype=int)
    valid_idx = np.asarray(valid_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    difficulty_arr = None
    bayes_margin_arr = None
    meta_dict: Dict[str, Array] = {}
    if metadata_frame is not None and not metadata_frame.empty:
        meta_frame = metadata_frame.reset_index(drop=True).copy()
        for col in meta_frame.columns:
            values = meta_frame[col].to_numpy()
            if col == "difficulty_score":
                difficulty_arr = values.astype(float)
            elif col == "bayes_margin":
                bayes_margin_arr = values.astype(float)
            meta_dict[col] = values

    preprocessor = _build_preprocessor(raw_features)
    X_train = preprocessor.fit_transform(raw_features.iloc[train_idx])
    X_valid = preprocessor.transform(raw_features.iloc[valid_idx])
    X_test = preprocessor.transform(raw_features.iloc[test_idx])

    X_all = np.zeros((raw_features.shape[0], X_train.shape[1]), dtype=float)
    X_all[train_idx] = np.asarray(X_train, dtype=float)
    X_all[valid_idx] = np.asarray(X_valid, dtype=float)
    X_all[test_idx] = np.asarray(X_test, dtype=float)

    metadata = {
        "preprocessor": preprocessor,
        "raw_feature_columns": raw_features.columns.astype(str).tolist(),
        "split_sizes": {
            "train": int(train_idx.shape[0]),
            "valid": int(valid_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
    }

    return BinaryClassificationDataset(
        dataset_name=dataset_name,
        train=_make_split(X_all, y_arr, group_arr, sample_id_arr, train_idx, meta_dict, difficulty_arr, bayes_margin_arr),
        valid=_make_split(X_all, y_arr, group_arr, sample_id_arr, valid_idx, meta_dict, difficulty_arr, bayes_margin_arr),
        test=_make_split(X_all, y_arr, group_arr, sample_id_arr, test_idx, meta_dict, difficulty_arr, bayes_margin_arr),
        feature_names=_feature_names_from_preprocessor(preprocessor),
        group_names=sorted(pd.unique(group_arr).tolist()),
        metadata=metadata,
    )


def load_real_binary_classification_dataset(
    dataset_name: str,
    repeat_id: int = 0,
    group_definition: str = "auto",
    processed_root: Path | str = DEFAULT_REAL_PROCESSED_ROOT,
    split_root: Path | str = DEFAULT_REAL_SPLIT_ROOT,
    random_state: int = 0,
) -> BinaryClassificationDataset:
    processed_root = Path(processed_root)
    split_root = Path(split_root)

    frame, processed_manifest = _load_cleaned_table(dataset_name=dataset_name, processed_root=processed_root)
    split_manifest = _load_split_manifest(dataset_name=dataset_name, repeat_id=repeat_id, split_root=split_root)

    if "__sample_id__" not in frame.columns or "__target__" not in frame.columns:
        raise KeyError(
            f"Cleaned table for {dataset_name!r} must contain '__sample_id__' and '__target__' columns"
        )

    requested_group_definition = str(group_definition).strip().lower()
    resolved_group_definition = _resolve_group_definition(
        dataset_name=dataset_name,
        group_definition=requested_group_definition,
    )
    group = _group_from_rule(frame=frame, dataset_name=dataset_name, group_definition=resolved_group_definition)

    raw_features = frame.drop(columns=["__sample_id__", "__target__"]).copy()
    raw_features = _coerce_categorical_columns(raw_features, processed_manifest.get("categorical_columns", []))
    sample_id = frame["__sample_id__"].astype(str).to_numpy()
    y = frame["__target__"].to_numpy(dtype=int)
    metadata_frame = pd.DataFrame(
        {
            "group_label": group.astype(str).to_numpy(),
            "repeat_id": np.full(frame.shape[0], int(repeat_id), dtype=int),
        }
    )

    dataset = _assemble_from_fixed_splits(
        dataset_name=f"{dataset_name}_repeat{int(repeat_id):02d}_{resolved_group_definition}",
        raw_features=raw_features,
        y=y,
        group=group.astype(str).to_numpy(),
        sample_id=sample_id,
        train_idx=np.asarray(split_manifest["train_idx"], dtype=int),
        valid_idx=np.asarray(split_manifest["valid_idx"], dtype=int),
        test_idx=np.asarray(split_manifest["test_idx"], dtype=int),
        metadata_frame=metadata_frame,
    )

    metadata = dict(dataset.metadata)
    metadata["real_dataset_name"] = str(dataset_name)
    metadata["repeat_id"] = int(repeat_id)
    metadata["group_definition"] = resolved_group_definition
    metadata["requested_group_definition"] = requested_group_definition
    metadata["processed_manifest"] = processed_manifest
    metadata["split_manifest"] = split_manifest

    dataset = BinaryClassificationDataset(
        dataset_name=dataset.dataset_name,
        train=dataset.train,
        valid=dataset.valid,
        test=dataset.test,
        feature_names=list(dataset.feature_names),
        group_names=list(dataset.group_names),
        metadata=metadata,
    )

    if resolved_group_definition == "difficulty_group":
        difficulty_dataset = with_margin_based_difficulty_groups(dataset, random_state=random_state)
        difficulty_metadata = dict(difficulty_dataset.metadata)
        difficulty_metadata["real_dataset_name"] = str(dataset_name)
        difficulty_metadata["repeat_id"] = int(repeat_id)
        difficulty_metadata["group_definition"] = resolved_group_definition
        difficulty_metadata["requested_group_definition"] = requested_group_definition
        difficulty_metadata["processed_manifest"] = processed_manifest
        difficulty_metadata["split_manifest"] = split_manifest
        return BinaryClassificationDataset(
            dataset_name=difficulty_dataset.dataset_name,
            train=difficulty_dataset.train,
            valid=difficulty_dataset.valid,
            test=difficulty_dataset.test,
            feature_names=list(difficulty_dataset.feature_names),
            group_names=list(difficulty_dataset.group_names),
            metadata=difficulty_metadata,
        )

    return dataset
