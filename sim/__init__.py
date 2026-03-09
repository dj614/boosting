"""Simulation utilities for experiment 1."""

from .experiment1_data import (
    generate_dataset_bundle,
    generate_features,
    generate_latent_signal,
    make_oracle_metadata,
    summarize_dataset_bundle,
)
from .experiment1_eval import compute_metrics, subgroup_metrics
from .experiment1_analysis import (
    build_analysis_summary,
    make_error_variance_scatter,
    make_method_comparison_plot,
    make_pairwise_comparison_table,
)
from .experiment1_models import build_model, default_methods_for_task

__all__ = [
    "build_analysis_summary",
    "build_model",
    "compute_metrics",
    "default_methods_for_task",
    "make_error_variance_scatter",
    "make_method_comparison_plot",
    "make_pairwise_comparison_table",
    "generate_dataset_bundle",
    "generate_features",
    "generate_latent_signal",
    "make_oracle_metadata",
    "subgroup_metrics",
    "summarize_dataset_bundle",
]
