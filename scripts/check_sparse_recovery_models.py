#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util

alias_src = ROOT / "sim" / "group_risk_redistribution_analysis.py"
if alias_src.exists() and "sim.experiment1_step3_analysis" not in sys.modules:
    spec = importlib.util.spec_from_file_location("sim.experiment1_step3_analysis", alias_src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

import numpy as np

from sim.ctb_semantics import ctb_semantic_role, sparse_recovery_support_semantics
from sim.sparse_recovery_data import generate_sparse_regression_dataset
from sim.sparse_recovery_models import build_experiment4_model



def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanity-check experiment 4 model wrappers.")
    parser.add_argument("--design", choices=["independent", "block_correlated", "strong_collinear"], default="block_correlated")
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-valid", type=int, default=120)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--p", type=int, default=400)
    parser.add_argument("--s", type=int, default=10)
    parser.add_argument("--snr", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["l2boost", "bagged_componentwise", "ctb_sparse", "ctb_tree", "lasso"],
        choices=["l2boost", "bagged_componentwise", "ctb_sparse", "ctb_tree", "lasso", "xgb_tree"],
    )
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment4_model_check"))
    return parser



def _summarize_model(model_name: str, model, dataset) -> dict:
    test_pred = model.predict(dataset.test.X)
    test_mse = float(np.mean((dataset.test.y - test_pred) ** 2))
    support_hat = np.asarray(model.selected_support_, dtype=int)
    summary = {
        "model_name": model_name,
        "test_mse": test_mse,
        "selected_support_size": int(support_hat.shape[0]),
        "selected_support_head": support_hat[: min(10, support_hat.shape[0])].tolist(),
        "support_semantic_role": sparse_recovery_support_semantics(model_name),
        "ctb_semantic_role": ctb_semantic_role(model_name),
    }
    if hasattr(model, "feature_importances_") and getattr(model, "feature_importances_", None) is not None:
        summary["has_feature_importances"] = 1
    if hasattr(model, "selection_frequency_") and getattr(model, "selection_frequency_", None) is not None:
        summary["has_selection_frequency"] = 1
    if hasattr(model, "selected_step_") and getattr(model, "selected_step_", None) is not None:
        summary["selected_step"] = int(model.selected_step_)
    if hasattr(model, "selected_checkpoint_") and getattr(model, "selected_checkpoint_", None) is not None:
        summary["selected_checkpoint"] = int(model.selected_checkpoint_)
    if hasattr(model, "selected_alpha_") and getattr(model, "selected_alpha_", None) is not None:
        summary["selected_alpha"] = float(model.selected_alpha_)
    return summary



def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = generate_sparse_regression_dataset(
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        p=args.p,
        s=args.s,
        design=args.design,
        snr=args.snr,
        seed=args.seed,
    )

    summaries = []
    for model_name in args.models:
        model = build_experiment4_model(model_name=model_name, random_state=args.seed)
        model.fit(dataset.train, dataset.valid)
        summaries.append(_summarize_model(model_name=model_name, model=model, dataset=dataset))
        if getattr(model, "selection_trace_", None) is not None:
            model.selection_trace_.to_csv(outdir / f"{model_name}_selection_trace.csv", index=False)

    payload = {
        "config": dataset.config,
        "summaries": summaries,
    }
    (outdir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
