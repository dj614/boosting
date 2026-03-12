from __future__ import annotations

from typing import Dict, Iterable, List

from .schema import RealDatasetSpec


_REAL_DATASET_SPECS: Dict[str, RealDatasetSpec] = {
    "diabetes": RealDatasetSpec(
        canonical_name="diabetes",
        task_type="binary_classification",
        source_type="openml",
        openml_name="diabetes",
        openml_version=1,
        target_column="class",
        positive_aliases=("tested_positive", "positive", "1", "true", "yes"),
        positive_label="tested_positive",
        default_group_rules=("difficulty_group",),
        notes="Pima Indians Diabetes dataset from OpenML.",
    ),
    "german_numer": RealDatasetSpec(
        canonical_name="german_numer",
        task_type="binary_classification",
        source_type="openml",
        openml_name="German-Credit-Risk",
        openml_version=1,
        target_column="class",
        positive_aliases=("good", "1", "true", "yes"),
        positive_label="good",
        default_group_rules=("difficulty_group",),
        notes="German credit risk dataset in OpenML tabular form.",
    ),
    "credit": RealDatasetSpec(
        canonical_name="credit",
        task_type="binary_classification",
        source_type="openml",
        openml_name="Australian",
        openml_version=1,
        target_column="A15",
        positive_aliases=("1", "+", "positive", "true", "yes"),
        positive_label="1",
        default_group_rules=("difficulty_group",),
        notes="Australian Credit Approval dataset from OpenML.",
    ),
    "blood": RealDatasetSpec(
        canonical_name="blood",
        task_type="binary_classification",
        source_type="openml",
        openml_name="blood-transfusion-service-center",
        openml_version=1,
        target_column="Class",
        positive_aliases=("2",),
        positive_label="2",
        default_group_rules=("difficulty_group",),
        notes="Blood Transfusion Service Center dataset from OpenML.",
    ),
    "titanic": RealDatasetSpec(
        canonical_name="titanic",
        task_type="binary_classification",
        source_type="openml",
        openml_name="Titanic",
        openml_version=1,
        target_column="survived",
        positive_aliases=("1", "yes", "true", "survived"),
        positive_label="1",
        default_group_rules=("sex", "pclass", "sex_pclass", "difficulty_group"),
        notes="Original Titanic tabular dataset from OpenML.",
    ),
    "raisin": RealDatasetSpec(
        canonical_name="raisin",
        task_type="binary_classification",
        source_type="uci_archive",
        target_column="Class",
        positive_aliases=("kecimen", "1", "true", "yes"),
        positive_label="Kecimen",
        uci_download_url="https://archive.ics.uci.edu/static/public/850/raisin.zip",
        archive_member="Raisin_Dataset/Raisin_Dataset.xlsx",
        default_group_rules=("difficulty_group",),
        notes="Raisin dataset from the UCI Machine Learning Repository.",
    ),
    "qsar": RealDatasetSpec(
        canonical_name="qsar",
        task_type="binary_classification",
        source_type="openml",
        openml_name="qsar-biodeg",
        openml_version=1,
        target_column="Class",
        positive_aliases=("ready biodegradable", "rb", "1", "true", "yes"),
        positive_label="ready biodegradable",
        default_group_rules=("difficulty_group",),
        notes="QSAR biodegradation dataset from OpenML.",
    ),
    "climate": RealDatasetSpec(
        canonical_name="climate",
        task_type="binary_classification",
        source_type="openml",
        openml_name="climate-model-simulation-crashes",
        openml_version=1,
        target_column="class",
        positive_aliases=("1", "true", "yes", "positive"),
        positive_label="1",
        default_group_rules=("difficulty_group",),
        notes="Climate model simulation crashes dataset from OpenML.",
    ),
}



def list_real_dataset_names() -> List[str]:
    return sorted(_REAL_DATASET_SPECS.keys())



def iter_real_dataset_specs() -> Iterable[RealDatasetSpec]:
    for name in list_real_dataset_names():
        yield _REAL_DATASET_SPECS[name]



def get_real_dataset_spec(name: str) -> RealDatasetSpec:
    key = str(name).strip().lower()
    if key not in _REAL_DATASET_SPECS:
        raise KeyError(f"Unknown real dataset: {name!r}. Available: {', '.join(list_real_dataset_names())}")
    return _REAL_DATASET_SPECS[key]
