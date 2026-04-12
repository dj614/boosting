# CTB

This is an experimental repository for studying **bagging / boosting / XGBoost / CTB**, covering simulation studies, open tabular benchmarks, and comparisons between prediction and statistical inference.

## Environment

Python **3.10–3.12** is recommended.

```bash
pip install -r requirements.txt
```

## How to run

Run the following commands from the repository root:

```bash
bash scripts/open_tabular_all.sh
bash scripts/e1_instability_matching.sh
bash scripts/e2_group_risk_redistribution.sh
bash scripts/e3_prediction_vs_inference.sh
bash scripts/e4_sparse_recovery.sh
```

## Experiments

- **E1 instability matching**: compares how bagging / GBDT / CTB respond to different types of base-learner instability.
- **E2 group risk redistribution**: analyzes how overall risk is redistributed across different sample groups.
- **E3 prediction vs inference**: distinguishes between strong predictive performance and strong statistical inference.
- **E4 sparse recovery**: compares structure recovery ability in high-dimensional sparse models.
- **Open tabular benchmark**: provides a unified comparison of bagging, RF, GBDT, XGB, and CTB on multiple real-world classification and regression datasets.

## Main directories

```text
configs/         configuration files
data/            data generation and data structures
metrics/         evaluation metrics
models/          baseline models
real_data/       real classification data processing
real_regression/ real regression data processing
runners/         benchmark entry points
scripts/         one-click experiment scripts
sim/             core CTB and simulation implementations
plots/           plotting utilities
```

## Outputs

Results are written to `outputs/` by default, typically including:

- per-seed result tables
- summary CSV / JSON files
- analysis plots
- optional wandb logs

## Dependencies

Core dependencies include: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `PyYAML`, `seaborn`, and `wandb`.
