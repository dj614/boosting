#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_data.schema import DEFAULT_REAL_DATA_ROOT, DEFAULT_REAL_PROCESSED_ROOT, DEFAULT_REAL_SPLIT_ROOT  # noqa: E402
from real_regression.schema import (  # noqa: E402
    DEFAULT_REAL_REGRESSION_DATA_ROOT,
    DEFAULT_REAL_REGRESSION_PROCESSED_ROOT,
    DEFAULT_REAL_REGRESSION_SPLIT_ROOT,
)
from runners.open_tabular_benchmark import (  # noqa: E402
    DEFAULT_FAMILIES,
    DEFAULT_SELECTION_CHECKPOINTS,
    default_classification_datasets,
    default_regression_datasets,
    run_open_tabular_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified open tabular benchmark for classification and regression.")
    parser.add_argument("--classification-datasets", nargs="*", default=default_classification_datasets())
    parser.add_argument("--regression-datasets", nargs="*", default=default_regression_datasets())
    parser.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--selection-checkpoints", nargs="*", type=int, default=DEFAULT_SELECTION_CHECKPOINTS)
    parser.add_argument("--max-depths", nargs="*", type=int, default=[1, 3, 5])
    parser.add_argument("--min-samples-leafs", nargs="*", type=int, default=[1, 5])
    parser.add_argument("--learning-rates", nargs="*", type=float, default=[0.03, 0.1])
    parser.add_argument("--subsamples", nargs="*", type=float, default=[0.7, 1.0])
    parser.add_argument("--colsample-bytree", nargs="*", type=float, default=[0.8])
    parser.add_argument("--ctb-inner-bootstraps", nargs="*", type=int, default=[4, 8])
    parser.add_argument("--ctb-etas", nargs="*", type=float, default=[0.5, 1.0])
    parser.add_argument("--ctb-instability-penalty", type=float, default=0.0)
    parser.add_argument("--ctb-weight-power", type=float, default=1.0)
    parser.add_argument("--ctb-weight-eps", type=float, default=1e-8)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--classification-raw-root", type=Path, default=DEFAULT_REAL_DATA_ROOT)
    parser.add_argument("--classification-processed-root", type=Path, default=DEFAULT_REAL_PROCESSED_ROOT)
    parser.add_argument("--classification-split-root", type=Path, default=DEFAULT_REAL_SPLIT_ROOT)
    parser.add_argument("--regression-raw-root", type=Path, default=DEFAULT_REAL_REGRESSION_DATA_ROOT)
    parser.add_argument("--regression-processed-root", type=Path, default=DEFAULT_REAL_REGRESSION_PROCESSED_ROOT)
    parser.add_argument("--regression-split-root", type=Path, default=DEFAULT_REAL_REGRESSION_SPLIT_ROOT)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/open_tabular_benchmark"))
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_open_tabular_benchmark(
        classification_datasets=args.classification_datasets,
        regression_datasets=args.regression_datasets,
        families=args.families,
        max_rounds=args.max_rounds,
        selection_checkpoints=args.selection_checkpoints,
        max_depths=args.max_depths,
        min_samples_leafs=args.min_samples_leafs,
        learning_rates=args.learning_rates,
        subsamples=args.subsamples,
        colsample_bytree=args.colsample_bytree,
        ctb_inner_bootstraps=args.ctb_inner_bootstraps,
        ctb_etas=args.ctb_etas,
        ctb_instability_penalty=args.ctb_instability_penalty,
        ctb_weight_power=args.ctb_weight_power,
        ctb_weight_eps=args.ctb_weight_eps,
        n_repeats=args.n_repeats,
        base_seed=args.base_seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        classification_raw_root=args.classification_raw_root,
        classification_processed_root=args.classification_processed_root,
        classification_split_root=args.classification_split_root,
        regression_raw_root=args.regression_raw_root,
        regression_processed_root=args.regression_processed_root,
        regression_split_root=args.regression_split_root,
        output_root=args.outdir,
        n_jobs=args.n_jobs,
    )
    print(f"[done] summary_test_metrics: {summary['summary_test_metrics_path']}")
    print(f"[done] summary_valid_selection: {summary['summary_valid_selection_path']}")
    if int(summary.get("n_errors", 0)) > 0:
        print(f"[warning] encountered {summary['n_errors']} runs with errors; see errors.csv under {summary['output_root']}")


if __name__ == "__main__":
    main()
