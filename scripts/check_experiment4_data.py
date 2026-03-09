#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim.experiment4_data import (
    generate_sparse_regression_dataset,
    summarize_sparse_regression_dataset,
    top_correlated_features,
)



def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and sanity-check experiment 4 sparse regression data.")
    parser.add_argument("--design", choices=["independent", "block_correlated", "strong_collinear"], default="independent")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-valid", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--p", type=int, default=2000)
    parser.add_argument("--s", type=int, default=10)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--beta-scale", type=float, default=1.0)
    parser.add_argument("--beta-pattern", choices=["equal", "decay", "mixed_sign"], default="equal")
    parser.add_argument("--support-strategy", choices=["first", "spaced", "random"], default="spaced")
    parser.add_argument("--snr", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corr-heatmap-size", type=int, default=80)
    parser.add_argument("--top-k-neighbors", type=int, default=5)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment4_data_check"))
    return parser



def _plot_beta_stem(beta_true: np.ndarray, support_true: np.ndarray, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    markerline, stemlines, baseline = ax.stem(np.arange(beta_true.shape[0]), beta_true)
    plt.setp(markerline, markersize=3)
    plt.setp(stemlines, linewidth=1.0)
    plt.setp(baseline, linewidth=0.8)
    ax.scatter(support_true, beta_true[support_true], s=18, zorder=3)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("beta_true")
    ax.set_title("True sparse coefficient vector")
    fig.tight_layout()
    fig.savefig(outdir / "beta_true_stem.png", dpi=160)
    plt.close(fig)



def _plot_correlation_heatmap(X: np.ndarray, max_features: int, outdir: Path) -> None:
    k = min(max_features, X.shape[1])
    corr = np.corrcoef(X[:, :k], rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(corr, aspect="auto", interpolation="nearest")
    ax.set_title(f"Train correlation heatmap (first {k} features)")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "train_corr_heatmap.png", dpi=160)
    plt.close(fig)



def _plot_signal_vs_response(signal: np.ndarray, y: np.ndarray, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(signal, y, s=10, alpha=0.45)
    ax.set_xlabel("signal = X beta")
    ax.set_ylabel("y")
    ax.set_title("Signal-response sanity check")
    fig.tight_layout()
    fig.savefig(outdir / "signal_vs_y.png", dpi=160)
    plt.close(fig)



def _write_neighbor_table(dataset: object, top_k_neighbors: int, outdir: Path) -> None:
    neighbors = top_correlated_features(dataset, top_k=top_k_neighbors)
    rows = []
    for row_id, active_idx in enumerate(neighbors["active_idx"]):
        row = {"active_idx": int(active_idx)}
        for rank in range(top_k_neighbors):
            row[f"neighbor_{rank+1}_idx"] = int(neighbors["neighbor_idx"][row_id, rank])
            row[f"neighbor_{rank+1}_corr"] = float(neighbors["neighbor_corr"][row_id, rank])
        rows.append(row)
    (outdir / "active_feature_neighbors.json").write_text(json.dumps(rows, indent=2))



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
        rho=args.rho,
        block_size=args.block_size,
        beta_scale=args.beta_scale,
        beta_pattern=args.beta_pattern,
        support_strategy=args.support_strategy,
        snr=args.snr,
        seed=args.seed,
    )

    summary = summarize_sparse_regression_dataset(dataset)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "config.json").write_text(json.dumps(dataset.config, indent=2))
    (outdir / "support_true.json").write_text(json.dumps(dataset.support_true.tolist(), indent=2))

    _plot_beta_stem(dataset.beta_true, dataset.support_true, outdir)
    _plot_correlation_heatmap(dataset.train.X, max_features=args.corr_heatmap_size, outdir=outdir)
    _plot_signal_vs_response(dataset.train.signal, dataset.train.y, outdir)
    _write_neighbor_table(dataset, top_k_neighbors=args.top_k_neighbors, outdir=outdir)

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
