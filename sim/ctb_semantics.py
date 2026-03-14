from __future__ import annotations

from typing import Iterable, List


_CTB_FAMILY_ALIASES = {
    "ctb": "ctb",
    "ctb_tree": "ctb",
    "consensus_transport_boosting": "ctb",
    "consensus_transport_boosting_tree": "ctb",
}


def _clean_name(name: object) -> str:
    return str(name).strip().lower().replace("-", "_")


def _format_curvature_eps(value: float) -> str:
    return f"{float(value):.0e}"


def normalize_ctb_tree_family_name(name: object) -> str:
    cleaned = _clean_name(name)
    if cleaned in _CTB_FAMILY_ALIASES:
        return _CTB_FAMILY_ALIASES[cleaned]
    return cleaned


def is_ctb_tree_family_name(name: object) -> bool:
    return normalize_ctb_tree_family_name(name) == "ctb"


def normalize_ctb_tree_family_sequence(names: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for name in names:
        canonical = normalize_ctb_tree_family_name(name)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def ctb_tree_model_name(
    *,
    depth: int,
    task_type: str = "regression",
    update_target_mode: str = "legacy",
    transport_curvature_eps: float = 1e-6,
    weak_learner_backend: str = "sklearn_tree",
    include_task_suffix: bool = True,
) -> str:
    depth_i = int(depth)
    if depth_i <= 0:
        raise ValueError("depth must be positive")
    base = f"ctb_depth{depth_i}"
    base = f"{base}__mode-{str(update_target_mode).strip()}__curv-{_format_curvature_eps(float(transport_curvature_eps))}"
    backend = _clean_name(weak_learner_backend)
    if backend != "sklearn_tree":
        base = f"{base}__wl-{backend}"
    if include_task_suffix:
        base = f"{base}_{str(task_type).strip().lower()}"
    return base


def ctb_tree_method_aliases(*, depth: int, task_type: str) -> List[str]:
    depth_i = int(depth)
    task = str(task_type).strip().lower()
    aliases: List[str] = []
    if depth_i == 1:
        aliases.extend(
            [
                f"ctb_stump_{task}",
                f"ctb_depth1_{task}",
                f"ctb_tree_stump_{task}",
                f"ctb_tree_depth1_{task}",
            ]
        )
    else:
        aliases.extend(
            [
                f"ctb_depth{depth_i}_{task}",
                f"ctb_tree_depth{depth_i}_{task}",
            ]
        )
    out: List[str] = []
    seen = set()
    for name in aliases:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def normalize_ctb_tree_method_name(name: object) -> str:
    cleaned = _clean_name(name)
    if not cleaned.startswith("ctb"):
        return cleaned

    if cleaned.startswith("ctb_tree_"):
        cleaned = "ctb_" + cleaned[len("ctb_tree_") :]

    if cleaned.startswith("ctb_depth1_"):
        cleaned = "ctb_stump_" + cleaned[len("ctb_depth1_") :]
    return cleaned


def canonical_ctb_tree_result_method(
    method_name: object,
    *,
    update_target_mode: str = "legacy",
    transport_curvature_eps: float = 1e-6,
) -> str:
    canonical = normalize_ctb_tree_method_name(method_name)
    if not canonical.startswith("ctb_"):
        return canonical
    return f"{canonical}__mode-{str(update_target_mode).strip()}__curv-{_format_curvature_eps(float(transport_curvature_eps))}"


def sparse_recovery_support_semantics(name: object) -> str:
    cleaned = _clean_name(name)
    if cleaned in {"l2boost", "bagged_componentwise", "lasso", "ctb_sparse"}:
        return "native_support"
    if cleaned.startswith("ctb_tree") or cleaned.startswith("ctb_depth") or cleaned == "xgb_tree":
        return "topk_importance"
    return "unknown"


def sparse_recovery_family_semantic_bucket(name: object) -> str:
    cleaned = _clean_name(name)
    if cleaned == "ctb_sparse":
        return "ctb_structural"
    if cleaned.startswith("ctb_tree") or cleaned.startswith("ctb_depth"):
        return "ctb_predictive_tree"
    if cleaned in {"l2boost", "bagged_componentwise", "lasso"}:
        return "explicit_support"
    if cleaned == "xgb_tree":
        return "tree_importance"
    return "other"


def ctb_semantic_role(name: object) -> str:
    cleaned = _clean_name(name)
    if cleaned == "ctb_sparse":
        return "ctb_sparse_structural"
    if cleaned.startswith("ctb_tree") or cleaned.startswith("ctb_depth") or cleaned.startswith("ctb_stump"):
        return "ctb_tree_predictive"
    return "non_ctb"
