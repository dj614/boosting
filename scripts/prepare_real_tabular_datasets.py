#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from real_data import (
    DEFAULT_REAL_DATA_ROOT,
    DEFAULT_REAL_PROCESSED_ROOT,
    list_real_dataset_names,
    prepare_real_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned real tabular datasets for downstream experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_real_dataset_names(),
        help="Dataset names to prepare. Defaults to all registered real datasets.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_REAL_DATA_ROOT,
        help="Directory containing downloaded raw real datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_REAL_PROCESSED_ROOT,
        help="Directory where cleaned real datasets will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset_name in args.datasets:
        path = prepare_real_dataset(dataset_name=dataset_name, raw_root=args.raw_root, output_root=args.output_root)
        print(f"[prepared] {dataset_name}: {path}")


if __name__ == "__main__":
