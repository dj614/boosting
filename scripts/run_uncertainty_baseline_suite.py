#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners import load_yaml_config, run_uncertainty_baseline_suite


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the experiment-3 prediction/UQ and grouped inference baseline suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/uncertainty_baseline_suite.yaml"))
    parser.add_argument("--output-root", type=Path, default=None)
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    config = load_yaml_config(args.config)
    output_root = args.output_root or Path(config.get("output_root", "outputs/experiment3_prediction_vs_inference"))
    summary = run_uncertainty_baseline_suite(config=config, output_root=output_root)
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to: {output_root}")


if __name__ == "__main__":
    main()
