#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


SUMMARY_JSON_NAMES = {"artifact_summary.json", "summary.json", "analysis_summary.json", "run_config.json"}
TABLE_PRIORITY = [
    "metrics_summary_aggregated.csv",
    "metrics_summary.csv",
    "group_metrics_summary.csv",
    "trial_metrics.csv",
    "dataset_summaries.csv",
    "model_summary_analysis.csv",
    "group_summary_analysis.csv",
    "pairwise_seed_comparisons.csv",
    "trajectory_core_aggregated.csv",
    "trajectory_group_aggregated.csv",
    "model_summary.csv",
    "stability_summary.csv",
    "merged_summary.csv",
    "predictive_summary.csv",
    "structural_summary.csv",
    "per_seed_results.csv",
    "summary.csv",
]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SKIP_TABLE_SUBSTRINGS = (
    "pointwise",
    "predictions_",
    "test_predictions",
    "trajectory_samples",
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log experiment outputs to Weights & Biases.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--job-type", type=str, default=None)
    parser.add_argument("--tags", type=str, default="")
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--max-table-rows", type=int, default=5000)
    parser.add_argument("--max-images", type=int, default=48)
    parser.add_argument("--log-artifact", action="store_true")
    return parser


def _clean_key(text: object) -> str:
    return str(text).replace("\\", "/").replace(" ", "_").replace("-", "_").replace(".", "/")


def _flatten_dict(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=full_key))
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in value):
                flat[_clean_key(full_key)] = ", ".join(str(x) for x in value)
            continue
        if isinstance(value, bool):
            flat[_clean_key(full_key)] = value
            continue
        if isinstance(value, int):
            flat[_clean_key(full_key)] = int(value)
            continue
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                continue
            flat[_clean_key(full_key)] = float(value)
            continue
        if isinstance(value, str):
            flat[_clean_key(full_key)] = value
    return flat


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_table(path: Path, max_rows: int) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if max_rows > 0 and len(frame) > max_rows:
        return frame.head(max_rows).copy()
    return frame


def _discover_config(output_dir: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for rel_path in [Path("run_config.json"), Path("config_snapshot.yaml"), Path("config_snapshot.yml")]:
        path = output_dir / rel_path
        if not path.exists():
            continue
        payload = _read_json(path) if path.suffix == ".json" else _read_yaml(path)
        if payload:
            config[rel_path.stem] = payload
    return config


def _discover_summary_scalars(output_dir: Path) -> Dict[str, Any]:
    scalars: Dict[str, Any] = {}
    for path in sorted(output_dir.rglob("*.json")):
        if path.name not in SUMMARY_JSON_NAMES:
            continue
        rel_key = _clean_key(path.relative_to(output_dir).with_suffix(""))
        scalars.update(_flatten_dict(_read_json(path), prefix=rel_key))

    for path in sorted(output_dir.rglob("summary.csv")):
        frame = _read_table(path, max_rows=1)
        if frame.empty:
            continue
        rel_key = _clean_key(path.relative_to(output_dir).with_suffix(""))
        scalars.update(_flatten_dict(frame.iloc[0].to_dict(), prefix=rel_key))
    return scalars


def _rank_table_path(path: Path) -> tuple[int, int, str]:
    try:
        priority = TABLE_PRIORITY.index(path.name)
    except ValueError:
        priority = len(TABLE_PRIORITY)
    return (priority, len(path.parts), str(path))


def _select_table_paths(output_dir: Path) -> List[Path]:
    selected: List[Path] = []
    seen: set[Path] = set()
    for path in sorted(output_dir.rglob("*.csv"), key=_rank_table_path):
        rel_str = str(path.relative_to(output_dir))
        if any(token in rel_str for token in SKIP_TABLE_SUBSTRINGS):
            continue
        if path.name in TABLE_PRIORITY or (path.relative_to(output_dir).parts and path.relative_to(output_dir).parts[0] == "analysis"):
            selected.append(path)
            seen.add(path)
    for path in sorted(output_dir.rglob("*.csv"), key=_rank_table_path):
        if path in seen:
            continue
        if path.name in {"trajectory_core.csv", "trajectory_groups.csv", "selection_trace.csv"}:
            selected.append(path)
            seen.add(path)
    return selected


def _select_image_paths(output_dir: Path, max_images: int) -> List[Path]:
    images = [path for path in sorted(output_dir.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS]
    return images[: max(0, max_images)]


def _relative_media_key(output_dir: Path, path: Path) -> str:
    return _clean_key(path.relative_to(output_dir).with_suffix(""))


def main() -> None:
    args = _make_parser().parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "wandb is not installed. Install it with `pip install wandb` or disable WANDB_ENABLE."
        ) from exc

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        group=args.group,
        job_type=args.job_type,
        tags=tags,
        notes=args.notes,
        mode=args.mode,
        config=_discover_config(output_dir) or None,
        reinit=True,
    )
    assert run is not None

    summary_scalars = _discover_summary_scalars(output_dir)
    if summary_scalars:
        run.summary.update(summary_scalars)
        wandb.log(summary_scalars)

    manifest: Dict[str, Any] = {"output_dir": str(output_dir), "logged_tables": [], "logged_images": []}

    for table_path in _select_table_paths(output_dir):
        frame = _read_table(table_path, max_rows=args.max_table_rows)
        if frame.empty:
            continue
        wandb.log({_relative_media_key(output_dir, table_path): wandb.Table(dataframe=frame)})
        manifest["logged_tables"].append(str(table_path.relative_to(output_dir)))

    for image_path in _select_image_paths(output_dir, max_images=args.max_images):
        wandb.log({_relative_media_key(output_dir, image_path): wandb.Image(str(image_path))})
        manifest["logged_images"].append(str(image_path.relative_to(output_dir)))

    if args.log_artifact:
        artifact_name = f"{run.project}-{run.id}-outputs"
        artifact = wandb.Artifact(name=artifact_name, type="experiment_outputs")
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                artifact.add_file(str(file_path), name=str(file_path.relative_to(output_dir)))
        run.log_artifact(artifact)
        manifest["artifact_name"] = artifact_name

    manifest_path = output_dir / "wandb_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    run.summary["wandb_manifest_path"] = str(manifest_path)
    run.finish()
    print(json.dumps(manifest, indent=2))
    print(f"Logged outputs from {output_dir} to wandb.")


if __name__ == "__main__":
    main()
