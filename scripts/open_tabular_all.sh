python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 4 --outdir outputs/open_tabular_benchmark_loss_report --use-report-metric-for-selection --ctb-target-modes loss_aware

python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_loss_report

python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 4 --outdir outputs/open_tabular_benchmark_loss_logloss --ctb-target-modes loss_aware

python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_loss_logloss

python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 4 --outdir outputs/open_tabular_benchmark_legacy_report --use-report-metric-for-selection

python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_legacy_report

python scripts/run_open_tabular_benchmark.py --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic --regression-datasets california_housing concrete_compressive_strength superconductivity diamonds --n-jobs 4 --outdir outputs/open_tabular_benchmark_legacy_logloss

python scripts/analyze_open_tabular_benchmark.py --input-dir outputs/open_tabular_benchmark_legacy_logloss
