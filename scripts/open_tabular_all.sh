# # classification

python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_classification_loss_report --use-report-metric-for-selection --max-rounds 300 --selection-checkpoints 100 200 300 --ctb-inner-bootstraps 4 --ctb-etas 1.0 --ctb-leaf-ridges 1.0 --families ctb

python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_classification_loss_report --task-types classification

# python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_classification_loss_logloss --ctb-leaf-ridges 1.0

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_classification_loss_logloss --task-types classification

# python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_classification_ctb_report --use-report-metric-for-selection --ctb-leaf-ridges 1.0

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_classification_ctb_report --task-types classification

# python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_classification_ctb_logloss --ctb-leaf-ridges 1.0

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_classification_ctb_logloss --task-types classification

# regression

python scripts/run_open_tabular_benchmark.py --classification-datasets   --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 12 --outdir outputs/open_tabular_benchmark_regression_loss_report --use-report-metric-for-selection --max-rounds 300 --selection-checkpoints 100 200 300 --ctb-inner-bootstraps 4 --ctb-etas 1.0 --ctb-leaf-ridges 1.0 --families ctb

python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_regression_loss_report  --task-types regression

# python scripts/run_open_tabular_benchmark.py --classification-datasets   --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 12 --outdir outputs/open_tabular_benchmark_regression_loss_logloss --ctb-leaf-ridges 1.0

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_regression_loss_logloss  --task-types regression

# python scripts/run_open_tabular_benchmark.py --classification-datasets   --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 12 --outdir outputs/open_tabular_benchmark_regression_ctb_report --use-report-metric-for-selection --ctb-leaf-ridges 1.0

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_regression_ctb_report  --task-types regression

# python scripts/run_open_tabular_benchmark.py --classification-datasets   --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 12 --outdir outputs/open_tabular_benchmark_regression_ctb_logloss --ctb-leaf-ridges 1.0

# python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_regression_ctb_logloss  --task-types regression

# # experiments

# bash scripts/e1_instability_matching.sh

# bash scripts/e2_group_risk_redistribution.sh

# bash scripts/e4_sparse_recovery.sh

# bash scripts/e3_prediction_vs_inference.sh
