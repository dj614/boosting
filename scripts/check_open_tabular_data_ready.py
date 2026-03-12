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
    default_classification_datasets,
    default_regression_datasets,
    ensure_open_tabular_data_ready,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, preprocess, split, and validate all open tabular benchmark datasets.")
    parser.add_argument("--classification-datasets", nargs="*", default=default_classification_datasets())
    parser.add_argument("--regression-datasets", nargs="*", default=default_regression_datasets())
    parser.add_argument("--n-repeats", type=int, default=3)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_open_tabular_data_ready(
        classification_datasets=args.classification_datasets,
        regression_datasets=args.regression_datasets,
        n_repeats=args.n_repeats,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        base_seed=args.base_seed,
        classification_raw_root=args.classification_raw_root,
        classification_processed_root=args.classification_processed_root,
        classification_split_root=args.classification_split_root,
        regression_raw_root=args.regression_raw_root,
        regression_processed_root=args.regression_processed_root,
        regression_split_root=args.regression_split_root,
    )
    print('[ok] all requested tabular datasets were downloaded, prepared, split, and reloaded successfully.')


if __name__ == "__main__":
    main()
