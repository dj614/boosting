#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from real_data import (
    DEFAULT_REAL_PROCESSED_ROOT,
    create_real_data_split_manifests,
    list_real_dataset_names,
)
from real_data.schema import DEFAULT_REAL_SPLIT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create repeated train/valid/test split manifests for real datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_real_dataset_names(),
        help="Dataset names to split. Defaults to all registered real datasets.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_REAL_PROCESSED_ROOT,
        help="Directory containing cleaned real datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_REAL_SPLIT_ROOT,
        help="Directory where split manifests will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = create_real_data_split_manifests(
        dataset_names=args.datasets,
        processed_root=args.processed_root,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        n_repeats=args.n_repeats,
        base_seed=args.base_seed,
    )
    for dataset_name, paths in outputs.items():
        for path in paths:
            print(f"[split] {dataset_name}: {path}")


if __name__ == "__main__":
    main()
