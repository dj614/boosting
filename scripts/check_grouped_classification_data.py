#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.linear_model import LogisticRegression

from sim.grouped_classification_data import (
    load_adult_income,
    simulate_grouped_classification,
    summarize_binary_classification_dataset,
    with_margin_based_difficulty_groups,
)
from sim.grouped_classification_eval import evaluate_binary_predictions, make_binary_prediction_frame, save_prediction_frame



def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanity-check step-1 data and metrics for experiment 1.")
    parser.add_argument("--dataset", choices=["simulated", "adult"], default="simulated")
    parser.add_argument(
        "--group-definition",
        choices=["sex", "age_bucket", "education_bucket", "sex_age", "difficulty"],
        default="sex",
        help="Only used for Adult. 'difficulty' means build attribute groups first, then relabel with train-only logistic margins.",
    )
    parser.add_argument("--n-samples", type=int, default=12000, help="Only used for simulated data.")
    parser.add_argument("--n-features", type=int, default=8, help="Only used for simulated data.")
    parser.add_argument("--valid-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiment1_step1_check"))
    return parser



def _load_dataset(args: argparse.Namespace):
    if args.dataset == "simulated":
        return simulate_grouped_classification(
            n_samples=args.n_samples,
            n_features=args.n_features,
            valid_size=args.valid_size,
            test_size=args.test_size,
            random_state=args.seed,
        )

    if args.group_definition == "difficulty":
        base = load_adult_income(
            group_definition="sex_age",
            valid_size=args.valid_size,
            test_size=args.test_size,
            random_state=args.seed,
        )
        return with_margin_based_difficulty_groups(base, random_state=args.seed)

    return load_adult_income(
        group_definition=args.group_definition,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.seed,
    )



def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(args)
    summary = summarize_binary_classification_dataset(dataset)
    (outdir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    model = LogisticRegression(max_iter=1000, random_state=args.seed)
    model.fit(dataset.train.X, dataset.train.y)
    test_prob = model.predict_proba(dataset.test.X)[:, 1]

    evaluation = evaluate_binary_predictions(y_true=dataset.test.y, y_prob=test_prob, group=dataset.test.group)
    eval_json = {
        "overall": evaluation["overall"],
        "core_risk": evaluation["core_risk"],
    }
    (outdir / "metrics.json").write_text(json.dumps(eval_json, indent=2))
    evaluation["group_metrics"].to_csv(outdir / "group_metrics.csv", index=False)

    prediction_frame = make_binary_prediction_frame(
        sample_id=dataset.test.sample_id,
        dataset_name=dataset.dataset_name,
        split="test",
        seed=args.seed,
        model_name="logistic_regression_baseline",
        group=dataset.test.group,
        y_true=dataset.test.y,
        y_prob=test_prob,
        metadata=dataset.test.metadata,
    )
    save_prediction_frame(prediction_frame, outdir / "predictions.csv")

    print(json.dumps(summary, indent=2))
    print(json.dumps(eval_json, indent=2))
    print(f"Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()
