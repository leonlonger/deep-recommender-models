"""Train a LightGBM CTR ranker for homepage recommendation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model import (
    FeatureEncoder,
    FeatureSpec,
    build_lgb_dataset,
    clean_metric_value,
    evaluate_binary_predictions,
    feature_importance_dataframe,
)

try:
    import yaml
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit("Missing dependency PyYAML. Install with: pip install -r requirements.txt") from exc


DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
PROJECT_ROOT = Path(__file__).resolve().parent
SUMMARY_METRICS = (
    "auc",
    "pcoc",
    "binary_logloss",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
)


@dataclass
class TrainingRunContext:
    run_id: str
    base_output_dir: Path
    output_dir: Path
    reproducibility_dir: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"), help="Path to training config YAML.")
    parser.add_argument("--inspect", action="store_true", help="Print dataset metadata and inferred features.")
    parser.add_argument("--limit", type=int, help="Limit rows read from each Parquet split for quick experiments.")
    parser.add_argument("--label-column", help="Override label column from preprocessed metadata.")
    parser.add_argument("--output-dir", help="Override training output directory.")
    parser.add_argument("--num-boost-round", type=int, help="Override LightGBM boosting rounds.")
    parser.add_argument("--skip-export", action="store_true", help="Train and evaluate without saving artifacts.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def resolve_path(path_value: Any, *, base_dir: Path | None = None) -> Path:
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    data_config = config.setdefault("data", {})
    training_config = config.setdefault("training", {})
    if args.limit is not None:
        data_config["row_limit"] = args.limit
    elif args.inspect:
        data_config["row_limit"] = int(data_config.get("inspect_row_limit", 1000))
    if args.output_dir:
        training_config["output_dir"] = args.output_dir
    if args.num_boost_round is not None:
        training_config["num_boost_round"] = args.num_boost_round


def resolve_preprocessed_data_dir(config: dict[str, Any], *, config_path: Path) -> Path:
    data_config = config.get("data", {})
    source = str(data_config.get("source", "preprocessed_parquet")).lower()
    if source not in {"preprocessed_parquet", "prepared_parquet"}:
        raise ValueError("homepage_lightgbm only supports data.source=preprocessed_parquet.")
    data_path = data_config.get("path")
    if not data_path:
        raise ValueError("data.path is required.")
    path = resolve_path(data_path, base_dir=config_path.parent)
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessed data directory not found: {path}. "
            "Generate it with homepage_recommendation/main.py --preprocess-output first."
        )
    if not path.is_dir():
        raise ValueError(f"data.path must be a preprocessed data directory: {path}")
    return path


def load_preprocessed_metadata(data_dir: Path) -> dict[str, Any]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Preprocessed metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def resolve_feature_spec(
    metadata: dict[str, Any],
    config: dict[str, Any],
    *,
    label_override: str | None,
) -> FeatureSpec:
    metadata_spec = metadata.get("feature_spec")
    if isinstance(metadata_spec, dict):
        feature_spec = FeatureSpec.from_dict(metadata_spec)
    else:
        label_config = config.get("label", {})
        feature_config = config.get("features", {})
        label_column = str(label_config.get("column") or "clicked")
        feature_spec = FeatureSpec(
            label_column=label_column,
            numeric_columns=tuple(str(column) for column in feature_config.get("numeric_columns", [])),
            categorical_columns=tuple(str(column) for column in feature_config.get("categorical_columns", [])),
        )
    if label_override:
        feature_spec = feature_spec.with_label(label_override)
    if not feature_spec.feature_columns:
        raise ValueError("No model features were selected in preprocessed metadata.")
    return feature_spec


def read_parquet_dataframe(
    path: Path,
    *,
    columns: list[str],
    row_limit: int | None,
) -> pd.DataFrame:
    if row_limit is None:
        return pd.read_parquet(path, columns=columns)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return pd.read_parquet(path, columns=columns).head(row_limit).copy()

    tables: list[Any] = []
    total_rows = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=min(max(row_limit, 1), 100_000), columns=columns):
        table = pa.Table.from_batches([batch])
        remaining = row_limit - total_rows
        if remaining <= 0:
            break
        if table.num_rows > remaining:
            table = table.slice(0, remaining)
        tables.append(table)
        total_rows += table.num_rows
        if total_rows >= row_limit:
            break

    if not tables:
        return pd.DataFrame(columns=columns)
    return pa.concat_tables(tables).to_pandas()


def coerce_binary_label(series: pd.Series, positive_threshold: float | None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        normalized = series.astype(str).str.strip().str.lower()
        mapped = normalized.map(
            {
                "true": 1.0,
                "t": 1.0,
                "yes": 1.0,
                "y": 1.0,
                "false": 0.0,
                "f": 0.0,
                "no": 0.0,
                "n": 0.0,
            }
        )
        numeric = numeric.fillna(mapped)

    if positive_threshold is not None:
        return (numeric > positive_threshold).astype("float32")

    numeric = numeric.astype("float32")
    valid = numeric.dropna()
    if valid.empty:
        return numeric
    label_min = float(valid.min())
    label_max = float(valid.max())
    if label_min < 0.0 or label_max > 1.0:
        raise ValueError(
            "Binary classification labels must be in [0, 1]. "
            "Set label.positive_threshold if the source target must be binarized."
        )
    return numeric


def coerce_configured_label(series: pd.Series, config: dict[str, Any]) -> pd.Series:
    positive_threshold = config.get("label", {}).get("positive_threshold")
    if positive_threshold is not None:
        positive_threshold = float(positive_threshold)
    return coerce_binary_label(series, positive_threshold)


def prepare_dataframe(
    dataframe: pd.DataFrame,
    feature_spec: FeatureSpec,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    missing = [column for column in feature_spec.required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Required columns are missing from preprocessed data: {missing}")

    prepared = dataframe.loc[:, feature_spec.required_columns].copy()
    prepared[feature_spec.label_column] = coerce_configured_label(prepared[feature_spec.label_column], config)
    before_drop = len(prepared)
    prepared = prepared.dropna(subset=[feature_spec.label_column]).reset_index(drop=True)
    dropped_rows = before_drop - len(prepared)
    if prepared.empty:
        raise ValueError("No training rows remain after dropping rows with missing labels.")
    return prepared, dropped_rows


def resolve_classification_threshold(config: dict[str, Any]) -> float:
    value = config.get("training", {}).get("classification_threshold", DEFAULT_CLASSIFICATION_THRESHOLD)
    if value is None:
        return DEFAULT_CLASSIFICATION_THRESHOLD
    threshold = float(value)
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("training.classification_threshold must be in the range (0, 1).")
    return threshold


def build_sample_weights(
    labels: pd.Series,
    mode: Any,
    *,
    threshold: float,
) -> np.ndarray | None:
    if str(mode).lower() != "balanced":
        return None
    binary_labels = (labels.to_numpy(dtype="float32") >= threshold).astype("int32")
    positives = float(binary_labels.sum())
    total = float(len(binary_labels))
    negatives = total - positives
    if positives == 0.0 or negatives == 0.0:
        return None
    negative_weight = total / (2.0 * negatives)
    positive_weight = total / (2.0 * positives)
    return np.where(binary_labels == 1, positive_weight, negative_weight).astype("float32")


def print_feature_summary(feature_spec: FeatureSpec, train_rows: int | None = None) -> None:
    print("Feature summary:")
    print(f"  label: {feature_spec.label_column}")
    print(f"  numeric ({len(feature_spec.numeric_columns)}): {list(feature_spec.numeric_columns)}")
    print(f"  categorical ({len(feature_spec.categorical_columns)}): {list(feature_spec.categorical_columns)}")
    if train_rows is not None:
        print(f"  train rows: {train_rows}")


def format_metric_value(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.6g}"


def print_evaluation_summary(evaluation: dict[str, dict[str, float]]) -> None:
    print("Evaluation summary:")
    for split_name in ("train", "validation"):
        split_metrics = evaluation.get(split_name)
        if not split_metrics:
            continue
        metric_text = ", ".join(
            f"{metric}={format_metric_value(split_metrics.get(metric))}"
            for metric in SUMMARY_METRICS
            if metric in split_metrics
        )
        print(f"  {split_name}: {metric_text}")


def sanitize_run_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    sanitized = sanitized.strip(".-")
    if not sanitized:
        raise ValueError("Training run_id cannot be empty after sanitization.")
    return sanitized


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def requested_run_id(training_config: dict[str, Any]) -> str | None:
    run_id = training_config.get("run_id")
    return str(run_id) if run_id else None


def unique_training_run_id(base_output_dir: Path, configured_run_id: str | None = None) -> str:
    base_run_id = sanitize_run_id(configured_run_id or default_run_id())
    runs_dir = base_output_dir / "runs"
    candidate = base_run_id
    suffix = 1
    while (runs_dir / candidate).exists():
        candidate = f"{base_run_id}-{suffix:02d}"
        suffix += 1
    return candidate


def create_training_run_context(training_config: dict[str, Any]) -> TrainingRunContext:
    base_output_dir = resolve_path(training_config.get("output_dir", "models/homepage_lightgbm"))
    run_id = unique_training_run_id(base_output_dir, requested_run_id(training_config))
    output_dir = base_output_dir / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    training_config["run_id"] = run_id
    training_config["base_output_dir"] = str(base_output_dir)
    training_config["resolved_output_dir"] = str(output_dir)

    print(f"Training run id: {run_id}")
    print(f"Training output dir: {output_dir}")
    return TrainingRunContext(
        run_id=run_id,
        base_output_dir=base_output_dir,
        output_dir=output_dir,
    )


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return vars(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json_file(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(clean_metric_value(value), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def run_reproducibility_command(
    command: list[str],
    *,
    cwd: Path = PROJECT_ROOT.parent,
    timeout: int = 60,
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }

    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def command_text(result: dict[str, Any]) -> str:
    sections = [
        f"$ {' '.join(str(part) for part in result['command'])}",
        f"returncode: {result['returncode']}",
    ]
    if result.get("stdout"):
        sections.extend(["", "stdout:", str(result["stdout"])])
    if result.get("stderr"):
        sections.extend(["", "stderr:", str(result["stderr"])])
    return "\n".join(sections).rstrip() + "\n"


def collect_environment_info() -> dict[str, Any]:
    package_versions: dict[str, Any] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", None),
    }
    try:
        import lightgbm as lgb

        package_versions["lightgbm"] = lgb.__version__
    except Exception as exc:  # pragma: no cover - diagnostic best effort
        package_versions["lightgbm_error"] = str(exc)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions,
    }


def write_environment_reproducibility_files(reproducibility_dir: Path) -> None:
    write_json_file(reproducibility_dir / "environment.json", collect_environment_info())
    pip_freeze = run_reproducibility_command([sys.executable, "-m", "pip", "freeze"], timeout=120)
    (reproducibility_dir / "pip_freeze.txt").write_text(
        pip_freeze["stdout"] if pip_freeze["returncode"] == 0 else command_text(pip_freeze),
        encoding="utf-8",
    )


def write_git_reproducibility_files(reproducibility_dir: Path) -> None:
    git_commands = {
        "top_level": ["git", "rev-parse", "--show-toplevel"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        "commit": ["git", "rev-parse", "HEAD"],
        "status_short": ["git", "status", "--short"],
    }
    git_info = {key: run_reproducibility_command(command) for key, command in git_commands.items()}
    write_json_file(reproducibility_dir / "git_info.json", git_info)

    status = run_reproducibility_command(["git", "status", "--short"])
    (reproducibility_dir / "git_status.txt").write_text(command_text(status), encoding="utf-8")

    staged_diff = run_reproducibility_command(["git", "diff", "--cached", "--binary", "--", "."])
    unstaged_diff = run_reproducibility_command(["git", "diff", "--binary", "--", "."])
    diff_text = (
        "# git diff --cached --binary -- .\n"
        f"{staged_diff['stdout'] or ''}\n"
        "# git diff --binary -- .\n"
        f"{unstaged_diff['stdout'] or ''}"
    )
    if staged_diff["returncode"] not in (0, None) or unstaged_diff["returncode"] not in (0, None):
        diff_text += "\n\n# diff command diagnostics\n"
        diff_text += command_text(staged_diff)
        diff_text += command_text(unstaged_diff)
    (reproducibility_dir / "git_diff.patch").write_text(diff_text, encoding="utf-8")


def path_metadata(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    metadata: dict[str, Any] = {
        "path": str(expanded),
        "exists": expanded.exists(),
    }
    if not expanded.exists():
        return metadata

    stat = expanded.stat()
    metadata.update(
        {
            "resolved_path": str(expanded.resolve()),
            "is_file": expanded.is_file(),
            "is_dir": expanded.is_dir(),
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
    )
    if expanded.is_file():
        metadata["size_bytes"] = stat.st_size
    return metadata


def collect_data_source_info(data_dir: Path) -> dict[str, Any]:
    paths = [
        data_dir,
        data_dir / "metadata.json",
        data_dir / "train.parquet",
        data_dir / "validation.parquet",
    ]
    return {
        "paths": [path_metadata(path) for path in paths],
        "note": "Training data files are recorded for reproducibility but are not copied.",
    }


def save_reproducibility_bundle(
    run_context: TrainingRunContext,
    *,
    config: dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
    data_dir: Path,
) -> Path:
    reproducibility_dir = run_context.output_dir / "reproducibility"
    reproducibility_dir.mkdir(parents=True, exist_ok=True)
    run_context.reproducibility_dir = reproducibility_dir

    source_config_path = config_path.expanduser()
    if source_config_path.exists():
        shutil.copy2(source_config_path, reproducibility_dir / "source_config.yaml")
    else:
        (reproducibility_dir / "source_config_missing.txt").write_text(
            f"Source config was not found: {source_config_path}\n",
            encoding="utf-8",
        )

    (reproducibility_dir / "effective_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    write_json_file(
        reproducibility_dir / "args.json",
        {
            "argv": sys.argv,
            "args": vars(args),
            "config_path": str(config_path),
            "cwd": os.getcwd(),
            "project_root": str(PROJECT_ROOT),
        },
    )
    write_json_file(reproducibility_dir / "data_sources.json", collect_data_source_info(data_dir))
    write_environment_reproducibility_files(reproducibility_dir)
    write_git_reproducibility_files(reproducibility_dir)

    print(f"Reproducibility bundle dir: {reproducibility_dir}")
    return reproducibility_dir


def lightgbm_params(config: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(config.get("model", {}))
    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 42))
    model_config.setdefault("objective", "binary")
    model_config.setdefault("metric", ["binary_logloss", "auc"])
    model_config.setdefault("boosting_type", "gbdt")
    model_config.setdefault("seed", seed)
    model_config.setdefault("feature_fraction_seed", seed)
    model_config.setdefault("bagging_seed", seed)
    model_config.setdefault("data_random_seed", seed)
    return model_config


def load_splits(
    data_dir: Path,
    feature_spec: FeatureSpec,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame | None, int]:
    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Preprocessed train split not found: {train_path}")

    row_limit = int(config.get("data", {}).get("row_limit")) if config.get("data", {}).get("row_limit") else None
    train_df = read_parquet_dataframe(
        train_path,
        columns=feature_spec.required_columns,
        row_limit=row_limit,
    )
    validation_df = None
    if validation_path.exists():
        validation_df = read_parquet_dataframe(
            validation_path,
            columns=feature_spec.required_columns,
            row_limit=row_limit,
        )
    return train_df, validation_df, row_limit or 0


def run_inspection(config: dict[str, Any], *, config_path: Path, args: argparse.Namespace) -> None:
    data_dir = resolve_preprocessed_data_dir(config, config_path=config_path)
    metadata = load_preprocessed_metadata(data_dir)
    feature_spec = resolve_feature_spec(metadata, config, label_override=args.label_column)
    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    print(f"Preprocessed data dir: {data_dir}")
    print(f"Train split: {train_path} ({'exists' if train_path.exists() else 'missing'})")
    print(f"Validation split: {validation_path} ({'exists' if validation_path.exists() else 'missing'})")
    print_feature_summary(feature_spec)
    stats = metadata.get("stats")
    if isinstance(stats, dict):
        print("Preprocessing stats:")
        print(json.dumps(clean_metric_value(stats), ensure_ascii=False, indent=2))


def run_training(config: dict[str, Any], *, config_path: Path, args: argparse.Namespace) -> None:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency lightgbm. Install with: pip install -r homepage_lightgbm/requirements.txt"
        ) from exc

    data_dir = resolve_preprocessed_data_dir(config, config_path=config_path)
    metadata = load_preprocessed_metadata(data_dir)
    feature_spec = resolve_feature_spec(metadata, config, label_override=args.label_column)
    train_df, validation_df, row_limit = load_splits(data_dir, feature_spec, config)

    if train_df.empty:
        raise ValueError("Loaded training split is empty.")
    train_df, dropped_train_rows = prepare_dataframe(train_df, feature_spec, config)
    if validation_df is not None:
        validation_df, dropped_validation_rows = prepare_dataframe(validation_df, feature_spec, config)
        if validation_df.empty:
            validation_df = None
    else:
        dropped_validation_rows = 0

    threshold = resolve_classification_threshold(config)
    print(f"Training from preprocessed dataset: {data_dir}")
    if row_limit:
        print(f"Row limit per split: {row_limit}")
    print_feature_summary(feature_spec, train_rows=len(train_df))
    if validation_df is not None:
        print(f"Validation rows: {len(validation_df)}")
    else:
        print("Validation disabled")
    if dropped_train_rows or dropped_validation_rows:
        print(f"Dropped rows with missing labels: train={dropped_train_rows}, validation={dropped_validation_rows}")

    encoder = FeatureEncoder.fit(train_df, feature_spec)
    train_features = encoder.transform(train_df)
    train_labels = train_df[feature_spec.label_column].to_numpy(dtype="float32")
    sample_weights = build_sample_weights(
        train_df[feature_spec.label_column],
        config.get("training", {}).get("class_weight"),
        threshold=threshold,
    )

    train_dataset = build_lgb_dataset(
        train_features,
        train_labels,
        weights=sample_weights,
        categorical_feature_names=encoder.categorical_feature_names,
    )
    valid_sets = [train_dataset]
    valid_names = ["train"]
    validation_features = None
    validation_labels = None
    if validation_df is not None:
        validation_features = encoder.transform(validation_df)
        validation_labels = validation_df[feature_spec.label_column].to_numpy(dtype="float32")
        validation_dataset = build_lgb_dataset(
            validation_features,
            validation_labels,
            categorical_feature_names=encoder.categorical_feature_names,
            reference=train_dataset,
        )
        valid_sets.append(validation_dataset)
        valid_names.append("validation")

    training_config = config.get("training", {})
    callbacks = [
        lgb.log_evaluation(period=int(training_config.get("log_evaluation_period", 50))),
    ]
    early_stopping_rounds = int(training_config.get("early_stopping_rounds", 50))
    if validation_df is not None and early_stopping_rounds > 0:
        callbacks.append(
            lgb.early_stopping(
                early_stopping_rounds,
                first_metric_only=bool(training_config.get("early_stopping_first_metric_only", True)),
            )
        )

    booster = lgb.train(
        lightgbm_params(config),
        train_dataset,
        num_boost_round=int(training_config.get("num_boost_round", 1000)),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    best_iteration = getattr(booster, "best_iteration", None) or None
    train_predictions = booster.predict(train_features, num_iteration=best_iteration)
    evaluation = {
        "train": evaluate_binary_predictions(
            train_labels,
            train_predictions,
            threshold=threshold,
        )
    }
    if validation_features is not None and validation_labels is not None:
        validation_predictions = booster.predict(validation_features, num_iteration=best_iteration)
        evaluation["validation"] = evaluate_binary_predictions(
            validation_labels,
            validation_predictions,
            threshold=threshold,
        )

    print_evaluation_summary(evaluation)
    print("Evaluation JSON:")
    print(json.dumps(clean_metric_value(evaluation), ensure_ascii=False, indent=2))

    if args.skip_export:
        print("Skipping model export because --skip-export was set.")
        return

    run_context = create_training_run_context(training_config)
    save_reproducibility_bundle(
        run_context,
        config=config,
        config_path=config_path,
        args=args,
        data_dir=data_dir,
    )

    model_path = run_context.output_dir / "model.txt"
    encoder_path = run_context.output_dir / "feature_encoder.json"
    metadata_path = run_context.output_dir / "training_metadata.json"
    importance_path = run_context.output_dir / "feature_importance.csv"

    booster.save_model(str(model_path), num_iteration=best_iteration)
    encoder.save(encoder_path)
    feature_importance_dataframe(booster).to_csv(importance_path, index=False)
    write_json_file(
        metadata_path,
        {
            "run_id": run_context.run_id,
            "output_dir": run_context.output_dir,
            "base_output_dir": run_context.base_output_dir,
            "reproducibility_dir": run_context.reproducibility_dir,
            "feature_spec": feature_spec.to_dict(),
            "best_iteration": best_iteration,
            "evaluation": evaluation,
            "config": config,
            "source_metadata": metadata,
        },
    )

    print("Artifacts written:")
    print(f"  model: {model_path}")
    print(f"  feature_encoder: {encoder_path}")
    print(f"  feature_importance: {importance_path}")
    print(f"  metadata: {metadata_path}")
    if run_context.reproducibility_dir is not None:
        print(f"  reproducibility: {run_context.reproducibility_dir}")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    apply_cli_overrides(config, args)

    if args.inspect:
        run_inspection(config, config_path=config_path, args=args)
        return

    run_training(config, config_path=config_path, args=args)


if __name__ == "__main__":
    main()
