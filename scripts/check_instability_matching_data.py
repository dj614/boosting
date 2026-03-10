#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from sim.instability_matching_data import generate_dataset_bundle, summarize_dataset_bundle


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and sanity-check experiment 1 synthetic data.")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--scenario", choices=["piecewise", "smooth", "pocket"], default="piecewise")
    parser.add_argument("--noise-type", choices=["homoscedastic", "heteroscedastic"], default="homoscedastic")
    parser.add_argument("--feature-dist", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-valid", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=5000)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment1_data_check"))
    return parser


def _plot_piecewise(bundle, outdir: Path) -> None:
    X = bundle.train.X
    f_true = bundle.train.f_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], f_true, s=10, alpha=0.5)
    ax.axvline(0.2, linestyle="--")
    ax.set_xlabel("X[:, 0]")
    ax.set_ylabel("f_true")
    ax.set_title("Piecewise sanity check")
    fig.tight_layout()
    fig.savefig(outdir / "piecewise_x0_vs_f.png", dpi=160)
    plt.close(fig)


def _plot_smooth(bundle, outdir: Path) -> None:
    X = bundle.train.X
    f_true = bundle.train.f_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X[:, 0], f_true, s=10, alpha=0.5)
    ax.set_xlabel("X[:, 0]")
    ax.set_ylabel("f_true")
    ax.set_title("Smooth sanity check")
    fig.tight_layout()
    fig.savefig(outdir / "smooth_x0_vs_f.png", dpi=160)
    plt.close(fig)


def _plot_pocket(bundle, outdir: Path) -> None:
    X = bundle.test.X
    pocket_strength = bundle.test.meta["pocket_strength"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    scatter = ax.scatter(X[:, 1], X[:, 2], c=pocket_strength, s=8, alpha=0.7)
    ax.set_xlabel("X[:, 1]")
    ax.set_ylabel("X[:, 2]")
    ax.set_title("Pocket strength on test split")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "pocket_heatmap.png", dpi=160)
    plt.close(fig)



def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = generate_dataset_bundle(
        task_type=args.task,
        scenario=args.scenario,
        n_train=args.n_train,
        n_valid=args.n_valid,
        n_test=args.n_test,
        p=args.p,
        feature_dist=args.feature_dist,
        noise_type=args.noise_type,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    summary = summarize_dataset_bundle(bundle)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.scenario == "piecewise":
        _plot_piecewise(bundle, outdir)
    elif args.scenario == "smooth":
        _plot_smooth(bundle, outdir)
    else:
        _plot_pocket(bundle, outdir)

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
