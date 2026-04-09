#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from real_data.catalog import list_real_dataset_names
from real_data.preprocess import materialize_real_dataset
from real_data.schema import DEFAULT_REAL_DATA_ROOT, DEFAULT_REAL_PROCESSED_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pos/neg label ratios for binary classification datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names. Default: all classification datasets in real_data.catalog.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_REAL_DATA_ROOT,
        help="Root directory of raw downloaded datasets.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_REAL_PROCESSED_ROOT,
        help="Root directory of processed datasets.",
    )
    parser.add_argument(
        "--as-csv",
        action="store_true",
        help="Print output as CSV instead of pretty table.",
    )
    return parser.parse_args()


def summarize_dataset(dataset_name: str, raw_root: Path, processed_root: Path) -> dict:
    frame, _ = materialize_real_dataset(
        dataset_name=dataset_name,
        raw_root=raw_root,
        output_root=processed_root,
    )

    if "__target__" not in frame.columns:
        raise KeyError(f"{dataset_name}: cleaned table does not contain '__target__'.")

    y = frame["__target__"].astype(int)
    n_total = int(len(y))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_total == 0:
        raise ValueError(f"{dataset_name}: empty dataset.")
    if n_pos + n_neg != n_total:
        raise ValueError(
            f"{dataset_name}: labels are not binary 0/1 after preprocessing."
        )

    return {
        "dataset": dataset_name,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_frac": n_pos / n_total,
        "neg_frac": n_neg / n_total,
        "pos_to_neg": (n_pos / n_neg) if n_neg > 0 else float("inf"),
    }


def main() -> None:
    args = parse_args()
    dataset_names = args.datasets if args.datasets else list_real_dataset_names()

    rows = []
    for dataset_name in dataset_names:
        rows.append(
            summarize_dataset(
                dataset_name=dataset_name,
                raw_root=args.raw_root,
                processed_root=args.processed_root,
            )
        )

    df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)

    if args.as_csv:
        print(df.to_csv(index=False))
    else:
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 200,
            "display.float_format", lambda x: f"{x:.6f}",
        ):
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()