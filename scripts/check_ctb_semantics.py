#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from sim.instability_matching_models import build_model
from sim.tabular_benchmark_models import TabularBenchmarkModelConfig, build_tabular_benchmark_wrapper
from sim.sparse_recovery_data import generate_sparse_regression_dataset
from sim.sparse_recovery_models import build_experiment4_model
from sim.ctb_semantics import (
    ctb_semantic_role,
    sparse_recovery_support_semantics,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test CTB semantics across e1/e2/e4.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/check_ctb_semantics"))
    return parser


class Split(SimpleNamespace):
    X: np.ndarray
    y: np.ndarray


def _classification_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 6))
    logits = 1.2 * X[:, 0] - 0.9 * X[:, 1] + 0.4 * X[:, 2]
    y = (rng.uniform(size=X.shape[0]) < (1.0 / (1.0 + np.exp(-logits)))).astype(int)
    return X, y


def _regression_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(80, 6))
    y = 1.5 * X[:, 0] - 0.75 * X[:, 1] + 0.25 * X[:, 2] + 0.1 * rng.normal(size=X.shape[0])
    return X, y


def _check_e1_aliases(seed: int) -> dict:
    Xr, yr = _regression_data(seed)
    Xc, yc = _classification_data(seed + 1)

    reg_base = build_model("ctb_stump_regression", task_type="regression", random_state=seed, ctb_n_estimators=12, ctb_inner_bootstraps=3)
    reg_alias = build_model("ctb_tree_stump_regression", task_type="regression", random_state=seed, ctb_n_estimators=12, ctb_inner_bootstraps=3)
    reg_base.fit(Xr, yr)
    reg_alias.fit(Xr, yr)
    reg_delta = float(np.max(np.abs(reg_base.predict(Xr[:16]) - reg_alias.predict(Xr[:16]))))

    cls_base = build_model("ctb_depth3_classification", task_type="classification", random_state=seed, ctb_n_estimators=12, ctb_inner_bootstraps=3)
    cls_alias = build_model("ctb_tree_depth3_classification", task_type="classification", random_state=seed, ctb_n_estimators=12, ctb_inner_bootstraps=3)
    cls_base.fit(Xc, yc)
    cls_alias.fit(Xc, yc)
    cls_delta = float(np.max(np.abs(cls_base.predict_proba(Xc[:16]) - cls_alias.predict_proba(Xc[:16]))))

    if reg_delta > 1e-10:
        raise AssertionError(f"e1 regression alias mismatch: {reg_delta}")
    if cls_delta > 1e-10:
        raise AssertionError(f"e1 classification alias mismatch: {cls_delta}")

    return {
        "regression_alias_max_abs_delta": reg_delta,
        "classification_alias_max_abs_delta": cls_delta,
    }


def _check_e2_wrappers(seed: int) -> dict:
    Xc, yc = _classification_data(seed + 2)
    Xr, yr = _regression_data(seed + 3)
    train_c = Split(X=Xc[:48], y=yc[:48])
    valid_c = Split(X=Xc[48:64], y=yc[48:64])
    test_c = Xc[64:]
    train_r = Split(X=Xr[:48], y=yr[:48])
    valid_r = Split(X=Xr[48:64], y=yr[48:64])
    test_r = Xr[64:]
    checkpoints = [4, 8, 12]

    cls_cfg = TabularBenchmarkModelConfig(
        task_type="classification",
        family="ctb_tree",
        max_depth=3,
        n_estimators=12,
        inner_bootstraps=3,
        eta=1.0,
        ctb_target_mode="loss_aware",
        ctb_curvature_eps=1e-3,
        random_state=seed,
    )
    cls_wrapper = build_tabular_benchmark_wrapper(cls_cfg, selection_checkpoints=checkpoints)
    cls_wrapper.fit(train_c, valid_c)
    cls_stage = cls_wrapper.predict_proba_staged(test_c, checkpoints=checkpoints)
    cls_prob = cls_wrapper.predict_proba(test_c)
    if cls_wrapper.selected_checkpoint_ not in checkpoints:
        raise AssertionError("classification selected checkpoint not in requested grid")
    if sorted(cls_stage.keys()) != checkpoints:
        raise AssertionError("classification staged checkpoints mismatch")
    if not np.all((cls_prob > 0.0) & (cls_prob < 1.0)):
        raise AssertionError("classification probabilities out of bounds")

    reg_cfg = TabularBenchmarkModelConfig(
        task_type="regression",
        family="ctb",
        max_depth=1,
        n_estimators=12,
        inner_bootstraps=3,
        eta=1.0,
        ctb_target_mode="loss_aware",
        ctb_curvature_eps=1e-3,
        random_state=seed,
    )
    reg_wrapper = build_tabular_benchmark_wrapper(reg_cfg, selection_checkpoints=checkpoints)
    reg_wrapper.fit(train_r, valid_r)
    reg_stage = reg_wrapper.predict_staged(test_r, checkpoints=checkpoints)
    reg_pred = reg_wrapper.predict(test_r)
    if reg_wrapper.selected_checkpoint_ not in checkpoints:
        raise AssertionError("regression selected checkpoint not in requested grid")
    if sorted(reg_stage.keys()) != checkpoints:
        raise AssertionError("regression staged checkpoints mismatch")
    if reg_pred.shape[0] != test_r.shape[0]:
        raise AssertionError("regression prediction shape mismatch")

    return {
        "classification_model_name": cls_wrapper.model_name,
        "classification_selected_checkpoint": int(cls_wrapper.selected_checkpoint_),
        "classification_selection_rows": int(0 if cls_wrapper.selection_trace_ is None else len(cls_wrapper.selection_trace_)),
        "regression_model_name": reg_wrapper.model_name,
        "regression_selected_checkpoint": int(reg_wrapper.selected_checkpoint_),
        "regression_selection_rows": int(0 if reg_wrapper.selection_trace_ is None else len(reg_wrapper.selection_trace_)),
    }


def _check_e4_support_semantics(seed: int) -> dict:
    dataset = generate_sparse_regression_dataset(
        n_train=60,
        n_valid=60,
        n_test=120,
        p=80,
        s=6,
        design="block_correlated",
        snr=4.0,
        seed=seed + 4,
    )
    sparse_model = build_experiment4_model(
        model_name="ctb_sparse",
        random_state=seed,
        max_steps=20,
        n_inner_bootstraps=3,
        selection_checkpoints=[5, 10, 20],
    )
    tree_model = build_experiment4_model(
        model_name="ctb_tree",
        random_state=seed,
        n_estimators=20,
        n_inner_bootstraps=3,
        max_depth=3,
        selection_checkpoints=[5, 10, 20],
    )
    sparse_model.fit(dataset.train, dataset.valid)
    tree_model.fit(dataset.train, dataset.valid)

    tree_support = np.asarray(tree_model.topk_support(dataset.support_true.shape[0]), dtype=int)
    sparse_support = np.asarray(sparse_model.selected_support_, dtype=int)
    if sparse_recovery_support_semantics("ctb_sparse") != "native_support":
        raise AssertionError("ctb_sparse semantic role mismatch")
    if sparse_recovery_support_semantics("ctb_tree") != "topk_importance":
        raise AssertionError("ctb_tree semantic role mismatch")
    if ctb_semantic_role("ctb_sparse") != "ctb_sparse_structural":
        raise AssertionError("ctb_sparse CTB role mismatch")
    if ctb_semantic_role("ctb_tree") != "ctb_tree_predictive":
        raise AssertionError("ctb_tree CTB role mismatch")
    if tree_support.ndim != 1 or sparse_support.ndim != 1:
        raise AssertionError("support outputs must be one-dimensional")

    return {
        "ctb_sparse_support_size": int(sparse_support.shape[0]),
        "ctb_tree_support_size": int(tree_support.shape[0]),
        "ctb_sparse_support_semantics": sparse_recovery_support_semantics("ctb_sparse"),
        "ctb_tree_support_semantics": sparse_recovery_support_semantics("ctb_tree"),
    }


def main() -> None:
    args = _make_parser().parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "seed": int(args.seed),
        "e1_alias_checks": _check_e1_aliases(args.seed),
        "e2_wrapper_checks": _check_e2_wrappers(args.seed),
        "e4_support_checks": _check_e4_support_semantics(args.seed),
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
