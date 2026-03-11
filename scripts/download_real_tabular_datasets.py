#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from real_data import DEFAULT_REAL_DATA_ROOT, download_real_dataset, list_real_dataset_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw real tabular datasets for benchmarking.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list_real_dataset_names(),
        help="Dataset names to download. Defaults to all registered real datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_REAL_DATA_ROOT,
        help="Directory where raw downloaded assets will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove any existing raw dataset directory before downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset_name in args.datasets:
        path = download_real_dataset(
            dataset_name=dataset_name,
            output_root=args.output_root,
            overwrite=bool(args.overwrite),
        )
        print(f"[downloaded] {dataset_name}: {path}")


if __name__ == "__main__":
    main()
