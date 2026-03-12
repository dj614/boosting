from __future__ import annotations

from typing import Dict, Iterable, List

from .schema import RealRegressionDatasetSpec

_REAL_REGRESSION_DATASET_SPECS: Dict[str, RealRegressionDatasetSpec] = {
    "acs_income_2018": RealRegressionDatasetSpec(
        canonical_name="acs_income_2018",
        task_type="regression",
        source_type="folktables",
        target_column="PINCP",
        folktables_dataset_name="ACSIncome",
        folktables_year=2018,
        feature_columns=("AGEP", "SCHL", "MAR", "RAC1P", "SEX", "WKHP", "COW"),
        categorical_columns=("SCHL", "MAR", "RAC1P", "SEX", "COW"),
        notes=(
            "Fixed ACS/Folktables regression benchmark for 2018 source microdata. "
            "Target is personal income (PINCP); features cover age, education, marital status, race, sex, work hours, and class of worker."
        ),
    ),
    "california_housing": RealRegressionDatasetSpec(
        canonical_name="california_housing",
        task_type="regression",
        source_type="sklearn",
        target_column="MedHouseVal",
        sklearn_dataset_name="fetch_california_housing",
        notes="California housing regression benchmark from scikit-learn.",
    ),
    "concrete_compressive_strength": RealRegressionDatasetSpec(
        canonical_name="concrete_compressive_strength",
        task_type="regression",
        source_type="uci",
        target_column="Concrete compressive strength(MPa, megapascals)",
        uci_dataset_id=165,
        notes="Concrete compressive strength dataset from the UCI repository.",
    ),
    "superconductivity": RealRegressionDatasetSpec(
        canonical_name="superconductivity",
        task_type="regression",
        source_type="uci",
        target_column="critical_temp",
        uci_dataset_id=464,
        notes="Superconductivity critical temperature dataset from the UCI repository.",
    ),
    "diamonds": RealRegressionDatasetSpec(
        canonical_name="diamonds",
        task_type="regression",
        source_type="manual_download",
        target_column="price",
        notes="Diamonds price regression benchmark based on the canonical seaborn dataset.",
    ),
}


def list_real_regression_dataset_names() -> List[str]:
    return sorted(_REAL_REGRESSION_DATASET_SPECS.keys())


def iter_real_regression_dataset_specs() -> Iterable[RealRegressionDatasetSpec]:
    for name in list_real_regression_dataset_names():
        yield _REAL_REGRESSION_DATASET_SPECS[name]


def get_real_regression_dataset_spec(name: str) -> RealRegressionDatasetSpec:
    key = str(name).strip().lower()
    if key not in _REAL_REGRESSION_DATASET_SPECS:
        raise KeyError(
            f"Unknown real regression dataset: {name!r}. "
            f"Available: {', '.join(list_real_regression_dataset_names())}"
        )
    return _REAL_REGRESSION_DATASET_SPECS[key]
