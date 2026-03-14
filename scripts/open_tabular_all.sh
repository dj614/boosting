# # classification

python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_classification_loss_report --use-report-metric-for-selection --ctb-target-modes loss_aware --families ctb

python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_classification_loss_report --task-types classification

# python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_classification_loss_logloss --ctb-target-modes loss_aware

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_classification_loss_logloss --task-types classification

# python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_classification_legacy_report --use-report-metric-for-selection

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_classification_legacy_report --task-types classification

# python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_classification_legacy_logloss

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_classification_legacy_logloss --task-types classification

# regression

python scripts/run_open_tabular_benchmark.py --classification-datasets   --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 12 --outdir outputs/open_tabular_benchmark_regression_loss_report --use-report-metric-for-selection --ctb-target-modes loss_aware --families ctb

python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_regression_loss_report  --task-types regression

# python scripts/run_open_tabular_benchmark.py --classification-datasets   --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 12 --outdir outputs/open_tabular_benchmark_regression_loss_logloss --ctb-target-modes loss_aware

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_regression_loss_logloss  --task-types regression

# python scripts/run_open_tabular_benchmark.py --classification-datasets   --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 12 --outdir outputs/open_tabular_benchmark_regression_legacy_report --use-report-metric-for-selection

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_regression_legacy_report  --task-types regression

# python scripts/run_open_tabular_benchmark.py --classification-datasets   --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 12 --outdir outputs/open_tabular_benchmark_regression_legacy_logloss

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_regression_legacy_logloss  --task-types regression

# # experiments

# bash scripts/e1_instability_matching.sh

# bash scripts/e2_group_risk_redistribution.sh

# bash scripts/e4_sparse_recovery.sh

# bash scripts/e3_prediction_vs_inference.sh
