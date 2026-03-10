from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_prediction_interval_width_plot(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = frame.copy()
    plot_frame["interval_length"] = plot_frame["upper"] - plot_frame["lower"]
    plot_frame = plot_frame.sort_values("sigma_true").reset_index(drop=True)

    plt.figure(figsize=(6, 4))
    plt.scatter(plot_frame["sigma_true"], plot_frame["interval_length"], s=10, alpha=0.35)
    plt.xlabel("True noise scale")
    plt.ylabel("Interval length")
    plt.title("Prediction interval width vs. true noise")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_grouped_inference_ci_plot(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_frame = frame.sort_values("seed").reset_index(drop=True)

    plt.figure(figsize=(6, 4))
    y_pos = range(plot_frame.shape[0])
    plt.errorbar(
        plot_frame["beta_hat"],
        list(y_pos),
        xerr=[plot_frame["beta_hat"] - plot_frame["ci_lower"], plot_frame["ci_upper"] - plot_frame["beta_hat"]],
        fmt="o",
        capsize=3,
    )
    if not plot_frame.empty:
        plt.axvline(float(plot_frame["beta_true"].iloc[0]), linestyle="--")
    plt.yticks(list(y_pos), [str(seed) for seed in plot_frame["seed"].tolist()])
    plt.xlabel("beta estimate")
    plt.ylabel("seed")
    plt.title("Grouped inference confidence intervals")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
