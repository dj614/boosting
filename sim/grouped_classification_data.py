from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # scikit-learn >= 1.2
    from sklearn.datasets import fetch_openml
except Exception:  # pragma: no cover
    fetch_openml = None


Array = np.ndarray


@dataclass(frozen=True)
class BinaryClassificationSplit:
    X: Array
    y: Array
    group: Array
    sample_id: Array
    difficulty_score: Optional[Array]
    bayes_margin: Optional[Array]
    metadata: Dict[str, Array]


@dataclass(frozen=True)
class BinaryClassificationDataset:
    dataset_name: str
    train: BinaryClassificationSplit
    valid: BinaryClassificationSplit
    test: BinaryClassificationSplit
    feature_names: List[str]
    group_names: List[str]
    metadata: Dict[str, object]


def _sigmoid(z: Array) -> Array:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))



def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)



def _as_numpy_strings(values: Sequence[object]) -> Array:
    return np.asarray([str(v) for v in values], dtype=object)



def _safe_stratify_labels(y: Sequence[int], group: Sequence[object]) -> Optional[Array]:
    y_arr = np.asarray(y)
    group_arr = _as_numpy_strings(group)
    joint = np.asarray([f"{g}__{int(label)}" for g, label in zip(group_arr, y_arr)], dtype=object)
    joint_counts = pd.Series(joint).value_counts(dropna=False)
    if not joint_counts.empty and int(joint_counts.min()) >= 2 and joint_counts.shape[0] > 1:
        return joint

    group_counts = pd.Series(group_arr).value_counts(dropna=False)
    if not group_counts.empty and int(group_counts.min()) >= 2 and group_counts.shape[0] > 1:
        return group_arr

    y_counts = pd.Series(y_arr).value_counts(dropna=False)
    if not y_counts.empty and int(y_counts.min()) >= 2 and y_counts.shape[0] > 1:
        return y_arr
    return None



def _split_indices(
    y: Array,
    group: Array,
    valid_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[Array, Array, Array]:
    if valid_size <= 0.0 or test_size <= 0.0 or valid_size + test_size >= 1.0:
        raise ValueError("valid_size and test_size must be > 0 and valid_size + test_size must be < 1")

    indices = np.arange(y.shape[0], dtype=int)
    stratify_all = _safe_stratify_labels(y=y, group=group)
    train_valid_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_all,
    )

    valid_fraction_within_train_valid = valid_size / (1.0 - test_size)
    stratify_train_valid = _safe_stratify_labels(y=y[train_valid_idx], group=group[train_valid_idx])
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_fraction_within_train_valid,
        random_state=random_state + 1,
        stratify=stratify_train_valid,
    )
    return np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)



def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [col for col in frame.columns if col not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were found")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)



def _feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    names = list(preprocessor.get_feature_names_out())
    cleaned = [name.replace("num__", "").replace("cat__", "") for name in names]
    return cleaned



def _slice_optional(array: Optional[Array], index: Array) -> Optional[Array]:
    if array is None:
        return None
    return np.asarray(array)[index]



def _slice_metadata(metadata: Mapping[str, Array], index: Array) -> Dict[str, Array]:
    out: Dict[str, Array] = {}
    for key, value in metadata.items():
        out[key] = np.asarray(value)[index]
    return out



def _make_split(
    X: Array,
    y: Array,
    group: Array,
    sample_id: Array,
    index: Array,
    metadata: Mapping[str, Array],
    difficulty_score: Optional[Array],
    bayes_margin: Optional[Array],
) -> BinaryClassificationSplit:
    return BinaryClassificationSplit(
        X=np.asarray(X[index], dtype=float),
        y=np.asarray(y[index], dtype=int),
        group=np.asarray(group[index], dtype=object),
        sample_id=np.asarray(sample_id[index], dtype=object),
        difficulty_score=_slice_optional(difficulty_score, index),
        bayes_margin=_slice_optional(bayes_margin, index),
        metadata=_slice_metadata(metadata, index),
    )



def _assemble_binary_dataset(
    dataset_name: str,
    raw_features: pd.DataFrame,
    y: Sequence[int],
    group: Sequence[object],
    sample_id: Sequence[object],
    metadata_frame: Optional[pd.DataFrame],
    valid_size: float,
    test_size: float,
    random_state: int,
) -> BinaryClassificationDataset:
    y_arr = np.asarray(y, dtype=int)
    group_arr = _as_numpy_strings(group)
    sample_id_arr = _as_numpy_strings(sample_id)

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

    train_idx, valid_idx, test_idx = _split_indices(
        y=y_arr,
        group=group_arr,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )

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



def _group_name_from_index(group_index: int) -> str:
    return {0: "A_easy_majority", 1: "B_medium", 2: "C_rare_hard"}[group_index]



def simulate_grouped_classification(
    n_samples: int = 12000,
    n_features: int = 8,
    group_probs: Sequence[float] = (0.70, 0.25, 0.05),
    valid_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 0,
) -> BinaryClassificationDataset:
    if n_features < 6:
        raise ValueError("n_features must be at least 6 for the default grouped simulator")
    group_probs_arr = np.asarray(group_probs, dtype=float)
    if group_probs_arr.shape[0] != 3 or not np.isclose(group_probs_arr.sum(), 1.0):
        raise ValueError("group_probs must contain 3 probabilities summing to 1")

    rng = np.random.default_rng(random_state)
    group_idx = rng.choice(3, size=n_samples, p=group_probs_arr)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

    X[group_idx == 0, 0] += 0.8
    X[group_idx == 0, 1] -= 0.5
    X[group_idx == 1, 2] += 0.4
    X[group_idx == 2, 0] *= 1.2
    X[group_idx == 2, 1] *= 1.2

    logit = np.zeros(n_samples, dtype=float)

    mask_a = group_idx == 0
    xa = X[mask_a]
    logit[mask_a] = 2.8 * (1.10 * xa[:, 0] - 0.95 * xa[:, 1] + 0.45 * xa[:, 2] - 0.25)

    mask_b = group_idx == 1
    xb = X[mask_b]
    logit[mask_b] = 1.20 * (
        1.10 * np.sin(1.50 * xb[:, 0])
        + 0.85 * (xb[:, 1] * xb[:, 2])
        - 0.70 * xb[:, 3]
        + 0.35 * xb[:, 4]
    )

    mask_c = group_idx == 2
    xc = X[mask_c]
    xor_term = np.where((xc[:, 0] > 0.0) ^ (xc[:, 1] > 0.0), 1.0, -1.0)
    pocket = np.exp(-((xc[:, 2] - 1.0) ** 2 + (xc[:, 3] + 0.7) ** 2) / 0.60)
    logit[mask_c] = 0.55 * (1.15 * xor_term + 1.60 * pocket - 0.45 * xc[:, 4] + 0.15 * xc[:, 5] - 0.10)

    prob = _sigmoid(logit)
    y = rng.binomial(1, prob, size=n_samples).astype(int)
    bayes_margin = np.abs(prob - 0.5)
    difficulty_score = 1.0 - 2.0 * bayes_margin

    feature_frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    group = np.asarray([_group_name_from_index(int(idx)) for idx in group_idx], dtype=object)
    sample_id = np.asarray([f"sim_{i:06d}" for i in range(n_samples)], dtype=object)
    metadata = pd.DataFrame(
        {
            "group_index": group_idx.astype(int),
            "group_label": group,
            "bayes_prob": prob.astype(float),
            "bayes_margin": bayes_margin.astype(float),
            "difficulty_score": difficulty_score.astype(float),
            "is_hard_group": (group_idx == 2).astype(int),
        }
    )

    return _assemble_binary_dataset(
        dataset_name="simulated_grouped_classification",
        raw_features=feature_frame,
        y=y,
        group=group,
        sample_id=sample_id,
        metadata_frame=metadata,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )



def _clean_adult_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for col in clean.columns:
        if clean[col].dtype == object:
            clean[col] = clean[col].astype(str).str.strip()
            clean[col] = clean[col].replace({"?": np.nan, "nan": np.nan})
    return clean



def _adult_group_series(frame: pd.DataFrame, group_definition: str) -> pd.Series:
    if group_definition == "sex":
        return frame["sex"].fillna("missing").astype(str)

    if group_definition == "age_bucket":
        age_bucket = pd.cut(
            frame["age"].astype(float),
            bins=[-np.inf, 35.0, 50.0, np.inf],
            labels=["age_18_35", "age_36_50", "age_51_plus"],
        )
        return age_bucket.astype(str)

    if group_definition == "education_bucket":
        edu_num = frame["education-num"].astype(float)
        edu_bucket = pd.cut(
            edu_num,
            bins=[-np.inf, 9.0, 12.0, np.inf],
            labels=["edu_hs_or_less", "edu_some_college", "edu_bachelors_plus"],
        )
        return edu_bucket.astype(str)

    if group_definition == "sex_age":
        sex = frame["sex"].fillna("missing").astype(str)
        age_bucket = _adult_group_series(frame, group_definition="age_bucket")
        return sex + "__" + age_bucket.astype(str)

    raise ValueError(
        "Unsupported group_definition. Expected one of: 'sex', 'age_bucket', 'education_bucket', 'sex_age'"
    )



def load_adult_income(
    group_definition: str = "sex",
    valid_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 0,
) -> BinaryClassificationDataset:
    if fetch_openml is None:  # pragma: no cover
        raise ImportError("scikit-learn fetch_openml is unavailable in this environment")

    bunch = fetch_openml(name="adult", version=2, as_frame=True)
    feature_frame = _clean_adult_frame(bunch.data)
    target = pd.Series(bunch.target).astype(str).str.strip().replace({">50K.": ">50K", "<=50K.": "<=50K"})
    y = (target == ">50K").astype(int).to_numpy(dtype=int)
    group = _adult_group_series(feature_frame, group_definition=group_definition)
    sample_id = np.asarray([f"adult_{i:06d}" for i in range(feature_frame.shape[0])], dtype=object)

    metadata = pd.DataFrame(
        {
            "raw_sex": feature_frame["sex"].astype(str).to_numpy(),
            "raw_age": feature_frame["age"].astype(float).to_numpy(),
            "group_label": group.astype(str).to_numpy(),
        }
    )
    if "education-num" in feature_frame.columns:
        metadata["education_num"] = feature_frame["education-num"].astype(float).to_numpy()

    return _assemble_binary_dataset(
        dataset_name=f"adult_income_{group_definition}",
        raw_features=feature_frame,
        y=y,
        group=group.astype(str).to_numpy(),
        sample_id=sample_id,
        metadata_frame=metadata,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )



def with_margin_based_difficulty_groups(
    dataset: BinaryClassificationDataset,
    n_bins: int = 3,
    random_state: int = 0,
) -> BinaryClassificationDataset:
    if n_bins != 3:
        raise ValueError("Only n_bins=3 is currently supported")

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(dataset.train.X, dataset.train.y)

    train_margin = np.abs(model.predict_proba(dataset.train.X)[:, 1] - 0.5)
    q1, q2 = np.quantile(train_margin, [1.0 / 3.0, 2.0 / 3.0])

    def _difficulty_labels(X: Array) -> Tuple[Array, Array]:
        margin = np.abs(model.predict_proba(X)[:, 1] - 0.5)
        labels = np.full(X.shape[0], "difficulty_easy", dtype=object)
        labels[margin <= q2] = "difficulty_medium"
        labels[margin <= q1] = "difficulty_hard"
        difficulty_score = 1.0 - 2.0 * margin
        return labels, difficulty_score

    train_group, train_difficulty = _difficulty_labels(dataset.train.X)
    valid_group, valid_difficulty = _difficulty_labels(dataset.valid.X)
    test_group, test_difficulty = _difficulty_labels(dataset.test.X)

    def _replace_split(split: BinaryClassificationSplit, group: Array, difficulty_score: Array) -> BinaryClassificationSplit:
        metadata = dict(split.metadata)
        metadata["attribute_group"] = split.group
        metadata["baseline_margin"] = np.abs(model.predict_proba(split.X)[:, 1] - 0.5)
        return replace(
            split,
            group=np.asarray(group, dtype=object),
            difficulty_score=np.asarray(difficulty_score, dtype=float),
            bayes_margin=None,
            metadata=metadata,
        )

    metadata = dict(dataset.metadata)
    metadata["difficulty_group_source"] = "logistic_regression_margin"
    metadata["difficulty_group_quantiles"] = {"q1": float(q1), "q2": float(q2)}

    return BinaryClassificationDataset(
        dataset_name=f"{dataset.dataset_name}_difficulty_groups",
        train=_replace_split(dataset.train, train_group, train_difficulty),
        valid=_replace_split(dataset.valid, valid_group, valid_difficulty),
        test=_replace_split(dataset.test, test_group, test_difficulty),
        feature_names=list(dataset.feature_names),
        group_names=["difficulty_easy", "difficulty_hard", "difficulty_medium"],
        metadata=metadata,
    )



def summarize_binary_classification_dataset(dataset: BinaryClassificationDataset) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "dataset_name": dataset.dataset_name,
        "n_features": int(len(dataset.feature_names)),
        "group_names": list(dataset.group_names),
    }

    for split_name in ["train", "valid", "test"]:
        split = getattr(dataset, split_name)
        summary[f"{split_name}_n"] = int(split.y.shape[0])
        summary[f"{split_name}_positive_rate"] = float(np.mean(split.y))
        group_counts = pd.Series(split.group).value_counts(dropna=False).sort_index()
        summary[f"{split_name}_group_counts"] = {str(k): int(v) for k, v in group_counts.items()}
        group_pos = (
            pd.DataFrame({"group": split.group, "y": split.y})
            .groupby("group", dropna=False)["y"]
            .mean()
            .sort_index()
        )
        summary[f"{split_name}_group_positive_rates"] = {str(k): float(v) for k, v in group_pos.items()}
        if split.difficulty_score is not None:
            summary[f"{split_name}_difficulty_score_mean"] = float(np.mean(split.difficulty_score))
        if split.bayes_margin is not None:
            summary[f"{split_name}_bayes_margin_mean"] = float(np.mean(split.bayes_margin))
    return summary
