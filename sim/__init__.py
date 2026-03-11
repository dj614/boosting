"""Simulation utilities for experiment 1."""

from .instability_matching_data import (
    generate_dataset_bundle,
    generate_features,
    generate_latent_signal,
    make_oracle_metadata,
    summarize_dataset_bundle,
)
from .instability_matching_eval import compute_metrics, subgroup_metrics
from .instability_matching_analysis import (
    build_analysis_summary,
    make_error_variance_scatter,
    make_method_comparison_plot,
    make_pairwise_comparison_table,
)
from .instability_matching_models import build_model, default_methods_for_task
from .grouped_classification_data import (
    BinaryClassificationDataset,
    BinaryClassificationSplit,
    load_adult_income,
    simulate_grouped_classification,
    summarize_binary_classification_dataset,
    with_margin_based_difficulty_groups,
)
from .grouped_classification_eval import (
    compute_binary_classification_metrics,
    compute_groupwise_binary_metrics,
    compute_risk_redistribution_metrics,
    evaluate_binary_predictions,
    hard_group_gain_vs_easy_group_sacrifice,
    make_binary_prediction_frame,
    save_prediction_frame,
)

try:
    from .experiment1_step3_analysis import (
        FocusPair,
        aggregate_trajectories,
        bootstrap_pairwise_metric_differences,
        infer_focus_pairs,
        load_step2_artifacts,
        make_analysis_summary,
        pairwise_loss_deltas,
        summarize_group_metrics,
        summarize_model_metrics,
    )
except ModuleNotFoundError:  # pragma: no cover
    from .group_risk_redistribution_analysis import (
        FocusPair,
        aggregate_trajectories,
        bootstrap_pairwise_metric_differences,
        infer_focus_pairs,
        load_step2_artifacts,
        make_analysis_summary,
        pairwise_loss_deltas,
        summarize_group_metrics,
        summarize_model_metrics,
    )

from .group_risk_ensemble_models import (
    BinaryEnsembleWrapper,
    EnsembleModelConfig,
    build_binary_ensemble_wrapper,
    expand_model_grid,
)

from .sparse_recovery_data import (
    SparseRegressionDataset,
    SparseRegressionSplit,
    generate_sparse_regression_dataset,
    summarize_sparse_regression_dataset,
    top_correlated_features,
)

from .sparse_recovery_eval import (
    aggregate_metric_table,
    make_feature_support_frame,
    regression_metrics as experiment4_regression_metrics,
    stability_selection_metrics,
    support_indicator,
    support_recovery_metrics,
)

from .sparse_recovery_models import (
    BaggedComponentwiseConfig,
    BaggedComponentwiseRegressorWrapper,
    L2BoostingConfig,
    L2BoostingRegressorWrapper,
    LassoPathConfig,
    LassoPathRegressorWrapper,
    XGBTreeConfig,
    XGBTreeRegressorWrapper,
    build_experiment4_model,
    default_experiment4_model_grid,
)

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
    "BinaryClassificationDataset",
    "BinaryClassificationSplit",
    "compute_binary_classification_metrics",
    "compute_groupwise_binary_metrics",
    "compute_risk_redistribution_metrics",
    "evaluate_binary_predictions",
    "hard_group_gain_vs_easy_group_sacrifice",
    "load_adult_income",
    "make_binary_prediction_frame",
    "save_prediction_frame",
    "BinaryEnsembleWrapper",
    "EnsembleModelConfig",
    "build_binary_ensemble_wrapper",
    "expand_model_grid",

    "SparseRegressionDataset",
    "SparseRegressionSplit",
    "generate_sparse_regression_dataset",
    "summarize_sparse_regression_dataset",
    "top_correlated_features",
    "BaggedComponentwiseConfig",
    "BaggedComponentwiseRegressorWrapper",
    "L2BoostingConfig",
    "L2BoostingRegressorWrapper",
    "LassoPathConfig",
    "LassoPathRegressorWrapper",
    "XGBTreeConfig",
    "XGBTreeRegressorWrapper",
    "build_experiment4_model",
    "default_experiment4_model_grid",
    "aggregate_metric_table",
    "make_feature_support_frame",
    "experiment4_regression_metrics",
    "stability_selection_metrics",
    "support_indicator",
    "support_recovery_metrics",
    "simulate_grouped_classification",
    "summarize_binary_classification_dataset",
    "with_margin_based_difficulty_groups",
    "FocusPair",
    "aggregate_trajectories",
    "bootstrap_pairwise_metric_differences",
    "infer_focus_pairs",
    "load_step2_artifacts",
    "make_analysis_summary",
    "pairwise_loss_deltas",
    "summarize_group_metrics",
    "summarize_model_metrics",
]
