python scripts/run_open_tabular_benchmark.py   --classification-datasets blood climate credit diabetes german_numer qsar raisin titanic   --regression-datasets   --n-jobs 12 --outdir outputs/open_tabular_benchmark_v1_classification

python scripts/analyze_open_tabular_benchmark.py   --input-dir outputs/open_tabular_benchmark_v1_classification   --task-types classification