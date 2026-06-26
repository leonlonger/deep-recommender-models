"""Train a TensorFlow DNN ranker for homepage recommendation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import tensorflow as tf

from model import (
    FeatureSpec,
    build_homepage_recommendation_model,
    build_numeric_normalizer,
    dataframe_to_dataset,
)

try:
    import yaml
except ImportError as exc:  # pragma: no cover - startup guard
    raise SystemExit("Missing dependency PyYAML. Install dependencies with: pip install -r requirements.txt") from exc


LABEL_CANDIDATES = (
    "label",
    "target",
    "clicked",
    "is_clicked",
    "click",
    "is_click",
    "engaged",
    "is_engaged",
    "converted",
    "is_converted",
    "conversion",
    "purchased",
    "is_purchased",
)

ID_LIKE_COLUMNS = {
    "id",
    "user",
    "user_id",
    "member_id",
    "customer_id",
    "visitor_id",
    "session_id",
    "item_id",
    "product_id",
    "content_id",
    "candidate_id",
    "sku",
    "brand_id",
    "category_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to training config YAML.")
    parser.add_argument("--limit", type=int, help="Override row limit for the configured data source.")
    parser.add_argument("--label-column", help="Override label column from config.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    parser.add_argument("--output-dir", help="Override training output directory.")
    parser.add_argument("--tensorboard-log-dir", help="Override TensorBoard base log directory.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream local Parquet training data in chunks instead of loading it all into memory.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable local Parquet streaming and use the in-memory pandas training path.",
    )
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Train without writing TensorBoard event logs.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Train and evaluate without saving model artifacts.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Load a small sample and print inferred schema without training.",
    )
    parser.add_argument(
        "--preprocess-output",
        help=(
            "Prepare the configured streaming Parquet data once and write train/validation "
            "Parquet files plus metadata to this directory, then exit."
        ),
    )
    parser.add_argument(
        "--overwrite-preprocessed",
        action="store_true",
        help="Allow --preprocess-output to replace existing preprocessed files in the output directory.",
    )
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


def as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]


def quote_bigquery_identifier(identifier: str) -> str:
    if "`" in identifier:
        raise ValueError(f"Backticks are not allowed in BigQuery identifiers: {identifier}")
    return f"`{identifier}`"


def build_bigquery_sql(data_config: dict[str, Any]) -> str:
    custom_query = data_config.get("query")
    if custom_query:
        return str(custom_query)

    table_id = data_config.get("table_id")
    if not table_id:
        raise ValueError("data.table_id is required for BigQuery source.")

    selected_columns = as_string_list(data_config.get("selected_columns"))
    if selected_columns:
        select_clause = ", ".join(quote_bigquery_identifier(column) for column in selected_columns)
    else:
        select_clause = "*"

    sql = f"SELECT {select_clause}\nFROM {quote_bigquery_identifier(str(table_id))}"
    where_clause = data_config.get("where") or data_config.get("where_clause")
    if where_clause:
        sql += f"\nWHERE {where_clause}"

    row_limit = data_config.get("row_limit")
    if row_limit:
        sql += f"\nLIMIT {int(row_limit)}"
    return sql


def get_bigquery_config(data_config: dict[str, Any]) -> dict[str, Any]:
    bigquery_config = data_config.get("bigquery")
    if isinstance(bigquery_config, dict):
        merged_config = dict(bigquery_config)
        if data_config.get("row_limit") is not None and merged_config.get("row_limit") is None:
            merged_config["row_limit"] = data_config["row_limit"]
        return merged_config
    return data_config


def load_bigquery_dataframe(data_config: dict[str, Any]) -> pd.DataFrame:
    try:
        from google.cloud import bigquery
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing dependency google-cloud-bigquery. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    project_id = data_config.get("project_id")
    client = bigquery.Client(project=project_id)
    sql = build_bigquery_sql(data_config)
    print("Running BigQuery query:")
    print(sql)
    return client.query(sql).result().to_dataframe()


def apply_row_limit(dataframe: pd.DataFrame, data_config: dict[str, Any]) -> pd.DataFrame:
    row_limit = data_config.get("row_limit")
    if row_limit:
        return dataframe.head(int(row_limit)).copy()
    return dataframe


def read_parquet_dataframe(path: Path, data_config: dict[str, Any]) -> pd.DataFrame:
    row_limit = data_config.get("row_limit")
    if not row_limit:
        return pd.read_parquet(path)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return pd.read_parquet(path).head(int(row_limit)).copy()

    limit = int(row_limit)
    tables: list[pa.Table] = []
    total_rows = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=min(limit, 100_000)):
        table = pa.Table.from_batches([batch])
        remaining = limit - total_rows
        if table.num_rows > remaining:
            table = table.slice(0, remaining)
        tables.append(table)
        total_rows += table.num_rows
        if total_rows >= limit:
            break

    if not tables:
        return pd.DataFrame()
    return pa.concat_tables(tables).to_pandas()


def load_local_dataframe(data_config: dict[str, Any], *, config_path: Path | None = None) -> pd.DataFrame:
    local_path = data_config.get("path")
    if not local_path:
        raise ValueError("data.path is required for local data sources.")

    base_dir = config_path.parent if config_path is not None else None
    path = resolve_path(local_path, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Local training data not found: {path}. "
            "Run scripts/dump_bigquery_data.py to dump the BigQuery table first."
        )

    source = str(data_config.get("source", "")).lower()
    suffix = path.suffix.lower()
    if source == "csv" or suffix == ".csv":
        dataframe = pd.read_csv(path)
    elif source in {"parquet", "local"} or suffix in {".parquet", ".pq"}:
        dataframe = read_parquet_dataframe(path, data_config)
    else:
        raise ValueError(
            f"Cannot infer local data format for {path}. "
            "Set data.source to parquet or csv."
        )
    return apply_row_limit(dataframe, data_config)


def resolve_local_data_path(data_config: dict[str, Any], *, config_path: Path | None = None) -> Path:
    local_path = data_config.get("path")
    if not local_path:
        raise ValueError("data.path is required for local data sources.")

    base_dir = config_path.parent if config_path is not None else None
    path = resolve_path(local_path, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Local training data not found: {path}. "
            "Run scripts/dump_bigquery_data.py to dump the BigQuery table first."
        )
    return path


def load_dataframe(config: dict[str, Any], *, config_path: Path | None = None) -> pd.DataFrame:
    data_config = config.get("data", {})
    source = str(data_config.get("source", "bigquery")).lower()
    if source == "bigquery":
        return load_bigquery_dataframe(get_bigquery_config(data_config))
    if source in {"csv", "parquet", "local"}:
        return load_local_dataframe(data_config, config_path=config_path)
    raise ValueError(f"Unsupported data.source: {source}")


def resolve_label_column(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    label_override: str | None,
) -> str:
    label_config = config.get("label", {})
    configured_columns = [
        *as_string_list(label_override),
        *as_string_list(label_config.get("column")),
        *as_string_list(label_config.get("fallback_columns")),
        *as_string_list(config.get("label_column")),
    ]
    for configured in configured_columns:
        if configured in dataframe.columns:
            return configured

    if configured_columns:
        raise ValueError(f"None of the configured label columns exist: {configured_columns}")

    for candidate in LABEL_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    candidates = ", ".join(LABEL_CANDIDATES)
    raise ValueError(
        "Could not infer label column. Set label.column in config.yaml or pass "
        f"--label-column. Tried: {candidates}"
    )


def looks_like_identifier(column: str) -> bool:
    lower = column.lower()
    return lower in ID_LIKE_COLUMNS or lower.endswith("_id") or lower.endswith("_uuid")


def validate_columns_exist(columns: list[str], dataframe: pd.DataFrame, group_name: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Configured {group_name} columns do not exist: {missing}")


def infer_feature_spec(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    label_column: str,
) -> FeatureSpec:
    features_config = config.get("features", {})
    explicit_numeric = as_string_list(features_config.get("numeric_columns"))
    explicit_categorical = as_string_list(features_config.get("categorical_columns"))
    exclude_columns = set(as_string_list(features_config.get("exclude_columns")))

    overlap = sorted(set(explicit_numeric) & set(explicit_categorical))
    if overlap:
        raise ValueError(f"Columns cannot be both numeric and categorical: {overlap}")

    validate_columns_exist(explicit_numeric, dataframe, "numeric")
    validate_columns_exist(explicit_categorical, dataframe, "categorical")

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    for column in explicit_numeric:
        if column != label_column and column not in numeric_columns:
            numeric_columns.append(column)
    for column in explicit_categorical:
        if column != label_column and column not in categorical_columns:
            categorical_columns.append(column)

    if features_config.get("auto_infer", True):
        selected = set(numeric_columns) | set(categorical_columns) | {label_column} | exclude_columns
        for column in dataframe.columns:
            if column in selected:
                continue

            dtype = dataframe[column].dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                continue
            if pd.api.types.is_bool_dtype(dtype) or looks_like_identifier(column):
                categorical_columns.append(column)
            elif pd.api.types.is_numeric_dtype(dtype):
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)

    return FeatureSpec(
        label_column=label_column,
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
    )


def feature_spec_from_dict(value: dict[str, Any]) -> FeatureSpec:
    return FeatureSpec(
        label_column=str(value["label_column"]),
        numeric_columns=tuple(str(column) for column in value.get("numeric_columns", [])),
        categorical_columns=tuple(str(column) for column in value.get("categorical_columns", [])),
    )


def feature_columns(feature_spec: FeatureSpec) -> list[str]:
    return [
        feature_spec.label_column,
        *feature_spec.numeric_columns,
        *feature_spec.categorical_columns,
    ]


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
            "If this table has counts, scores, ratings, or watch time as the target, "
            "set label.positive_threshold to binarize it."
        )
    return numeric


def is_missing_scalar(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def categorical_value_to_string(value: Any) -> str:
    if is_missing_scalar(value):
        return ""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return "|".join(categorical_value_to_string(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def coerce_categorical_series(series: pd.Series) -> pd.Series:
    values = series.astype("object")
    return values.map(categorical_value_to_string)


def prepare_dataframe(
    dataframe: pd.DataFrame,
    feature_spec: FeatureSpec,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    required_columns = [
        feature_spec.label_column,
        *feature_spec.numeric_columns,
        *feature_spec.categorical_columns,
    ]
    prepared = dataframe.loc[:, required_columns].copy()

    label_config = config.get("label", {})
    positive_threshold = label_config.get("positive_threshold")
    if positive_threshold is not None:
        positive_threshold = float(positive_threshold)

    prepared[feature_spec.label_column] = coerce_binary_label(
        prepared[feature_spec.label_column],
        positive_threshold=positive_threshold,
    )
    before_drop = len(prepared)
    prepared = prepared.dropna(subset=[feature_spec.label_column]).reset_index(drop=True)
    dropped_rows = before_drop - len(prepared)
    if prepared.empty:
        raise ValueError("No training rows remain after dropping rows with missing labels.")

    for column in feature_spec.numeric_columns:
        values = pd.to_numeric(prepared[column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prepared[column] = values.astype("float32")

    for column in feature_spec.categorical_columns:
        prepared[column] = coerce_categorical_series(prepared[column])

    return prepared, dropped_rows


def prepare_streaming_dataframe(
    dataframe: pd.DataFrame,
    feature_spec: FeatureSpec,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    try:
        return prepare_dataframe(dataframe, feature_spec, config)
    except ValueError as exc:
        if "No training rows remain after dropping rows with missing labels" in str(exc):
            return pd.DataFrame(), len(dataframe)
        raise


def split_dataframe(
    dataframe: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if validation_fraction <= 0.0 or len(dataframe) < 2:
        return dataframe.reset_index(drop=True), None

    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation_size = int(round(len(shuffled) * validation_fraction))
    validation_size = min(max(validation_size, 1), len(shuffled) - 1)
    validation_df = shuffled.iloc[:validation_size].reset_index(drop=True)
    train_df = shuffled.iloc[validation_size:].reset_index(drop=True)
    return train_df, validation_df


def build_class_weight(labels: pd.Series, mode: Any) -> dict[int, float] | None:
    if str(mode).lower() != "balanced":
        return None
    positives = float((labels >= 0.5).sum())
    total = float(len(labels))
    negatives = total - positives
    if positives == 0.0 or negatives == 0.0:
        return None
    return {
        0: total / (2.0 * negatives),
        1: total / (2.0 * positives),
    }


def build_balanced_class_weight_from_counts(
    positives: float,
    total: float,
    mode: Any,
) -> dict[int, float] | None:
    if str(mode).lower() != "balanced":
        return None
    negatives = total - positives
    if positives == 0.0 or negatives == 0.0:
        return None
    return {
        0: total / (2.0 * negatives),
        1: total / (2.0 * positives),
    }


def build_numeric_normalizer_from_stats(
    numeric_columns: tuple[str, ...],
    numeric_sum: np.ndarray,
    numeric_sum_squares: np.ndarray,
    count: int,
) -> tf.keras.layers.Layer | None:
    if not numeric_columns:
        return None
    if count <= 0:
        raise ValueError("Cannot build numeric normalizer without training rows.")

    mean = numeric_sum / float(count)
    variance = (numeric_sum_squares / float(count)) - np.square(mean)
    variance = np.maximum(variance, 1e-7)
    return tf.keras.layers.Normalization(
        axis=-1,
        mean=mean.astype("float32"),
        variance=variance.astype("float32"),
        name="numeric_normalization",
    )


def is_streaming_parquet_training(config: dict[str, Any], *, config_path: Path) -> bool:
    data_config = config.get("data", {})
    if not data_config.get("streaming", False):
        return False

    source = str(data_config.get("source", "")).lower()
    if source not in {"parquet", "local"}:
        return False

    path = resolve_local_data_path(data_config, config_path=config_path)
    return path.suffix.lower() in {".parquet", ".pq"}


def streaming_schema_sample(
    path: Path,
    data_config: dict[str, Any],
) -> pd.DataFrame:
    inspect_limit = int(data_config.get("inspect_row_limit", 1000))
    row_limit = data_config.get("row_limit")
    if row_limit:
        inspect_limit = min(inspect_limit, int(row_limit))

    sample_config = dict(data_config)
    sample_config["row_limit"] = max(inspect_limit, 1)
    return read_parquet_dataframe(path, sample_config)


def validation_mask_for_indices(
    row_indices: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> np.ndarray:
    if validation_fraction <= 0.0:
        return np.zeros(len(row_indices), dtype=bool)
    if validation_fraction >= 1.0:
        raise ValueError("training.validation_fraction must be less than 1.0.")

    # Deterministic row-level pseudo-random split that does not depend on chunking.
    mixed = (
        row_indices.astype("uint64") * np.uint64(6364136223846793005)
        + np.uint64(seed)
        + np.uint64(1442695040888963407)
    )
    thresholds = mixed.astype("float64") / float(np.iinfo("uint64").max)
    return thresholds < validation_fraction


def parquet_batches_to_dataframes(
    path: Path,
    columns: list[str],
    *,
    parquet_batch_rows: int,
    row_limit: int | None,
) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("Missing dependency pyarrow. Install dependencies with: pip install -r requirements.txt") from exc

    parquet_file = pq.ParquetFile(path)
    total_rows = 0
    for batch in parquet_file.iter_batches(batch_size=parquet_batch_rows, columns=columns):
        table = pa.Table.from_batches([batch])
        if row_limit is not None:
            remaining = row_limit - total_rows
            if remaining <= 0:
                break
            if table.num_rows > remaining:
                table = table.slice(0, remaining)

        start = total_rows
        total_rows += table.num_rows
        row_indices = np.arange(start, total_rows, dtype="uint64")
        yield table.to_pandas(), row_indices

        if row_limit is not None and total_rows >= row_limit:
            break


def parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("Missing dependency pyarrow. Install dependencies with: pip install -r requirements.txt") from exc

    return int(pq.ParquetFile(path).metadata.num_rows)


def write_preprocessed_frame(
    prepared: pd.DataFrame,
    *,
    split_name: str,
    output_dir: Path,
    writers: dict[str, Any],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("Missing dependency pyarrow. Install dependencies with: pip install -r requirements.txt") from exc

    table = pa.Table.from_pandas(prepared, preserve_index=False)
    writer = writers.get(split_name)
    if writer is None:
        writer = pq.ParquetWriter(
            output_dir / f"{split_name}.parquet",
            table.schema,
            compression="zstd",
        )
        writers[split_name] = writer
    writer.write_table(table)


def close_parquet_writers(writers: dict[str, Any]) -> None:
    for writer in writers.values():
        writer.close()


def serializable_preprocessing_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(stats["rows"]),
        "train_rows": int(stats["train_rows"]),
        "validation_rows": int(stats["validation_rows"]),
        "dropped_rows": int(stats["dropped_rows"]),
        "train_positives": float(stats["train_positives"]),
        "numeric_sum": [float(value) for value in stats["numeric_sum"]],
        "numeric_sum_squares": [float(value) for value in stats["numeric_sum_squares"]],
    }


def preprocessing_stats_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    stats = dict(metadata["stats"])
    stats["numeric_sum"] = np.array(stats.get("numeric_sum", []), dtype="float64")
    stats["numeric_sum_squares"] = np.array(stats.get("numeric_sum_squares", []), dtype="float64")
    return stats


def load_preprocessed_metadata(data_dir: Path) -> dict[str, Any]:
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Preprocessed metadata not found: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def is_preprocessed_parquet_training(config: dict[str, Any]) -> bool:
    data_config = config.get("data", {})
    source = str(data_config.get("source", "")).lower()
    return source in {"preprocessed_parquet", "prepared_parquet"} or bool(data_config.get("preprocessed"))


def resolve_preprocessed_data_dir(data_config: dict[str, Any], *, config_path: Path | None = None) -> Path:
    local_path = data_config.get("path")
    if not local_path:
        raise ValueError("data.path is required for preprocessed Parquet data.")
    base_dir = config_path.parent if config_path is not None else None
    path = resolve_path(local_path, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed data directory not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Preprocessed data.path must be a directory: {path}")
    return path


def build_preprocessed_training_config(
    config: dict[str, Any],
    *,
    output_dir: Path,
    feature_spec: FeatureSpec,
) -> dict[str, Any]:
    prepared_config = copy.deepcopy(config)
    data_config = prepared_config.setdefault("data", {})
    data_config.clear()
    data_config.update(
        {
            "source": "preprocessed_parquet",
            "path": str(output_dir),
            "streaming_batch_rows": int(config.get("data", {}).get("streaming_batch_rows", 100_000)),
        }
    )
    prepared_config.setdefault("label", {})["column"] = feature_spec.label_column
    prepared_config["features"] = {
        **prepared_config.get("features", {}),
        "auto_infer": False,
        "numeric_columns": list(feature_spec.numeric_columns),
        "categorical_columns": list(feature_spec.categorical_columns),
    }
    return prepared_config


def prepared_streaming_frames(
    path: Path,
    feature_spec: FeatureSpec,
    config: dict[str, Any],
    *,
    parquet_batch_rows: int,
    row_limit: int | None,
    validation_fraction: float,
    seed: int,
    split: str,
    shuffle: bool = False,
) -> Iterator[pd.DataFrame]:
    required_columns = [
        feature_spec.label_column,
        *feature_spec.numeric_columns,
        *feature_spec.categorical_columns,
    ]
    for batch_index, (dataframe, row_indices) in enumerate(
        parquet_batches_to_dataframes(
            path,
            required_columns,
            parquet_batch_rows=parquet_batch_rows,
            row_limit=row_limit,
        )
    ):
        if validation_fraction > 0.0:
            validation_mask = validation_mask_for_indices(
                row_indices,
                validation_fraction=validation_fraction,
                seed=seed,
            )
            if split == "train":
                dataframe = dataframe.loc[~validation_mask].reset_index(drop=True)
            elif split == "validation":
                dataframe = dataframe.loc[validation_mask].reset_index(drop=True)
            else:
                raise ValueError(f"Unsupported streaming split: {split}")

        if dataframe.empty:
            continue

        prepared, _ = prepare_streaming_dataframe(dataframe, feature_spec, config)
        if prepared.empty:
            continue
        if shuffle:
            prepared = prepared.sample(frac=1.0, random_state=seed + batch_index).reset_index(drop=True)
        yield prepared


def dataframe_to_batch_arrays(
    dataframe: pd.DataFrame,
    feature_spec: FeatureSpec,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for column in feature_spec.numeric_columns:
        features[column] = dataframe[column].to_numpy(dtype="float32").reshape(-1, 1)
    for column in feature_spec.categorical_columns:
        features[column] = dataframe[column].astype(str).to_numpy().reshape(-1, 1)
    labels = dataframe[feature_spec.label_column].to_numpy(dtype="float32").reshape(-1, 1)
    return features, labels


def streaming_output_signature(
    feature_spec: FeatureSpec,
    *,
    include_sample_weight: bool,
) -> Any:
    features_signature: dict[str, tf.TensorSpec] = {}
    for column in feature_spec.numeric_columns:
        features_signature[column] = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    for column in feature_spec.categorical_columns:
        features_signature[column] = tf.TensorSpec(shape=(None, 1), dtype=tf.string)
    label_signature = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
    if include_sample_weight:
        weight_signature = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        return features_signature, label_signature, weight_signature
    return features_signature, label_signature


def build_streaming_dataset(
    path: Path,
    feature_spec: FeatureSpec,
    config: dict[str, Any],
    *,
    parquet_batch_rows: int,
    row_limit: int | None,
    validation_fraction: float,
    seed: int,
    split: str,
    batch_size: int,
    shuffle: bool,
    class_weight: dict[int, float] | None = None,
) -> tf.data.Dataset:
    def generator() -> Iterator[Any]:
        for prepared in prepared_streaming_frames(
            path,
            feature_spec,
            config,
            parquet_batch_rows=parquet_batch_rows,
            row_limit=row_limit,
            validation_fraction=validation_fraction,
            seed=seed,
            split=split,
            shuffle=shuffle,
        ):
            for start in range(0, len(prepared), batch_size):
                batch = prepared.iloc[start : start + batch_size]
                features, labels = dataframe_to_batch_arrays(batch, feature_spec)
                if class_weight is not None:
                    weights = np.where(labels >= 0.5, class_weight[1], class_weight[0]).astype("float32")
                    yield features, labels, weights
                else:
                    yield features, labels

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=streaming_output_signature(
            feature_spec,
            include_sample_weight=class_weight is not None,
        ),
    ).prefetch(tf.data.AUTOTUNE)


def preprocessed_streaming_frames(
    path: Path,
    feature_spec: FeatureSpec,
    *,
    parquet_batch_rows: int,
    row_limit: int | None,
    shuffle: bool = False,
    seed: int,
) -> Iterator[pd.DataFrame]:
    for batch_index, (dataframe, _) in enumerate(
        parquet_batches_to_dataframes(
            path,
            feature_columns(feature_spec),
            parquet_batch_rows=parquet_batch_rows,
            row_limit=row_limit,
        )
    ):
        if dataframe.empty:
            continue
        if shuffle:
            dataframe = dataframe.sample(frac=1.0, random_state=seed + batch_index).reset_index(drop=True)
        yield dataframe


def build_preprocessed_dataset(
    path: Path,
    feature_spec: FeatureSpec,
    *,
    parquet_batch_rows: int,
    row_limit: int | None,
    seed: int,
    batch_size: int,
    shuffle: bool,
    class_weight: dict[int, float] | None = None,
) -> tf.data.Dataset:
    def generator() -> Iterator[Any]:
        for prepared in preprocessed_streaming_frames(
            path,
            feature_spec,
            parquet_batch_rows=parquet_batch_rows,
            row_limit=row_limit,
            shuffle=shuffle,
            seed=seed,
        ):
            for start in range(0, len(prepared), batch_size):
                batch = prepared.iloc[start : start + batch_size]
                features, labels = dataframe_to_batch_arrays(batch, feature_spec)
                if class_weight is not None:
                    weights = np.where(labels >= 0.5, class_weight[1], class_weight[0]).astype("float32")
                    yield features, labels, weights
                else:
                    yield features, labels

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=streaming_output_signature(
            feature_spec,
            include_sample_weight=class_weight is not None,
        ),
    ).prefetch(tf.data.AUTOTUNE)


def collect_preprocessed_training_stats(
    train_path: Path,
    validation_path: Path | None,
    feature_spec: FeatureSpec,
    *,
    parquet_batch_rows: int,
    row_limit: int | None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": 0,
        "train_rows": 0,
        "validation_rows": 0,
        "dropped_rows": 0,
        "train_positives": 0.0,
        "numeric_sum": np.zeros(len(feature_spec.numeric_columns), dtype="float64"),
        "numeric_sum_squares": np.zeros(len(feature_spec.numeric_columns), dtype="float64"),
    }

    for dataframe, _ in parquet_batches_to_dataframes(
        train_path,
        feature_columns(feature_spec),
        parquet_batch_rows=parquet_batch_rows,
        row_limit=row_limit,
    ):
        row_count = len(dataframe)
        stats["rows"] += row_count
        stats["train_rows"] += row_count
        labels = pd.to_numeric(dataframe[feature_spec.label_column], errors="coerce").fillna(0.0)
        stats["train_positives"] += float((labels >= 0.5).sum())
        if feature_spec.numeric_columns:
            numeric_values = dataframe.loc[:, feature_spec.numeric_columns].to_numpy(dtype="float64")
            stats["numeric_sum"] += numeric_values.sum(axis=0)
            stats["numeric_sum_squares"] += np.square(numeric_values).sum(axis=0)

    if validation_path is not None and validation_path.exists() and row_limit is None:
        stats["validation_rows"] = parquet_row_count(validation_path)
        stats["rows"] += stats["validation_rows"]
    elif validation_path is not None and validation_path.exists():
        stats["validation_rows"] = min(parquet_row_count(validation_path), int(row_limit))
        stats["rows"] += stats["validation_rows"]

    if stats["train_rows"] <= 0:
        raise ValueError("No preprocessed training rows remain.")
    return stats


def collect_streaming_training_stats(
    path: Path,
    feature_spec: FeatureSpec,
    config: dict[str, Any],
    *,
    parquet_batch_rows: int,
    row_limit: int | None,
    validation_fraction: float,
    seed: int,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "rows": 0,
        "train_rows": 0,
        "validation_rows": 0,
        "dropped_rows": 0,
        "train_positives": 0.0,
        "numeric_sum": np.zeros(len(feature_spec.numeric_columns), dtype="float64"),
        "numeric_sum_squares": np.zeros(len(feature_spec.numeric_columns), dtype="float64"),
    }

    required_columns = [
        feature_spec.label_column,
        *feature_spec.numeric_columns,
        *feature_spec.categorical_columns,
    ]
    for dataframe, row_indices in parquet_batches_to_dataframes(
        path,
        required_columns,
        parquet_batch_rows=parquet_batch_rows,
        row_limit=row_limit,
    ):
        stats["rows"] += len(dataframe)
        if validation_fraction > 0.0:
            validation_mask = validation_mask_for_indices(
                row_indices,
                validation_fraction=validation_fraction,
                seed=seed,
            )
        else:
            validation_mask = np.zeros(len(dataframe), dtype=bool)

        for split_name, mask in (
            ("train", ~validation_mask),
            ("validation", validation_mask),
        ):
            if not mask.any():
                continue
            prepared, dropped_rows = prepare_streaming_dataframe(
                dataframe.loc[mask].reset_index(drop=True),
                feature_spec,
                config,
            )
            stats["dropped_rows"] += dropped_rows
            if prepared.empty:
                continue

            row_count = len(prepared)
            if split_name == "train":
                stats["train_rows"] += row_count
                labels = prepared[feature_spec.label_column]
                stats["train_positives"] += float((labels >= 0.5).sum())
                if feature_spec.numeric_columns:
                    numeric_values = prepared.loc[:, feature_spec.numeric_columns].to_numpy(dtype="float64")
                    stats["numeric_sum"] += numeric_values.sum(axis=0)
                    stats["numeric_sum_squares"] += np.square(numeric_values).sum(axis=0)
            else:
                stats["validation_rows"] += row_count

    if stats["train_rows"] <= 0:
        raise ValueError("No streaming training rows remain after splitting and label cleanup.")
    return stats


def print_schema(dataframe: pd.DataFrame) -> None:
    print("Loaded columns:")
    for column, dtype in dataframe.dtypes.items():
        print(f"  - {column}: {dtype}")


def print_feature_summary(feature_spec: FeatureSpec, train_size: int | None = None) -> None:
    print("Feature summary:")
    print(f"  label: {feature_spec.label_column}")
    print(f"  numeric ({len(feature_spec.numeric_columns)}): {list(feature_spec.numeric_columns)}")
    print(f"  categorical ({len(feature_spec.categorical_columns)}): {list(feature_spec.categorical_columns)}")
    if train_size is not None:
        print(f"  rows: {train_size}")


def serializable_history(history: tf.keras.callbacks.History) -> dict[str, list[float]]:
    return {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }


def resolve_tensorboard_log_dir(
    tensorboard_config: dict[str, Any],
    output_dir: Path,
) -> Path:
    base_log_dir = resolve_path(
        tensorboard_config.get("log_dir", output_dir / "tensorboard"),
    )
    run_name = tensorboard_config.get("run_name")
    if not run_name or str(run_name).lower() == "auto":
        run_name = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return base_log_dir / str(run_name)


def build_tensorboard_callback(
    tensorboard_config: dict[str, Any],
    output_dir: Path,
) -> tf.keras.callbacks.TensorBoard | None:
    if not tensorboard_config.get("enabled", True):
        return None

    log_dir = resolve_tensorboard_log_dir(tensorboard_config, output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_config["resolved_log_dir"] = str(log_dir)
    print(f"TensorBoard log dir: {log_dir}")

    try:
        return tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=int(tensorboard_config.get("histogram_freq", 0)),
            write_graph=bool(tensorboard_config.get("write_graph", True)),
            write_images=bool(tensorboard_config.get("write_images", False)),
            update_freq=tensorboard_config.get("update_freq", "epoch"),
            profile_batch=tensorboard_config.get("profile_batch", 0),
        )
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing dependency tensorboard. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc


def export_model(
    model: tf.keras.Model,
    output_dir: Path,
    *,
    feature_spec: FeatureSpec,
    history: tf.keras.callbacks.History,
    evaluation: dict[str, float],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keras_path = output_dir / "final.keras"
    saved_model_dir = output_dir / "saved_model"
    metadata_path = output_dir / "training_metadata.json"

    model.save(str(keras_path))
    if hasattr(model, "export"):
        model.export(str(saved_model_dir))
    else:  # pragma: no cover - TensorFlow/Keras version compatibility
        tf.saved_model.save(model, str(saved_model_dir))

    metadata = {
        "feature_spec": feature_spec.to_dict(),
        "history": serializable_history(history),
        "evaluation": {metric: float(value) for metric, value in evaluation.items()},
        "config": config,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Artifacts written:")
    print(f"  keras_model: {keras_path}")
    print(f"  saved_model: {saved_model_dir}")
    print(f"  metadata: {metadata_path}")


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    data_config = config.setdefault("data", {})
    training_config = config.setdefault("training", {})
    tensorboard_config = training_config.setdefault("tensorboard", {})
    if args.limit is not None:
        data_config["row_limit"] = args.limit
    elif args.inspect:
        inspect_limit = int(data_config.get("inspect_row_limit", 1000))
        data_config["row_limit"] = inspect_limit
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.output_dir:
        training_config["output_dir"] = args.output_dir
    if args.tensorboard_log_dir:
        tensorboard_config["log_dir"] = args.tensorboard_log_dir
    if args.disable_tensorboard:
        tensorboard_config["enabled"] = False
    if args.streaming:
        data_config["streaming"] = True
    if args.no_streaming:
        data_config["streaming"] = False


def run_streaming_preprocessing(
    config: dict[str, Any],
    *,
    config_path: Path,
    args: argparse.Namespace,
) -> None:
    data_config = config.get("data", {})
    if not is_streaming_parquet_training(config, config_path=config_path):
        raise ValueError("--preprocess-output currently requires data.source parquet/local with data.streaming true.")

    source_path = resolve_local_data_path(data_config, config_path=config_path)
    output_dir = resolve_path(args.preprocess_output, base_dir=config_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    known_outputs = [
        output_dir / "train.parquet",
        output_dir / "validation.parquet",
        output_dir / "metadata.json",
        output_dir / "training_config.yaml",
    ]
    existing_outputs = [path for path in known_outputs if path.exists()]
    if existing_outputs and not args.overwrite_preprocessed:
        existing = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Preprocessed output already exists. Pass --overwrite-preprocessed to replace: "
            f"{existing}"
        )
    if args.overwrite_preprocessed:
        for path in existing_outputs:
            path.unlink()

    sample_dataframe = streaming_schema_sample(source_path, data_config)
    if sample_dataframe.empty:
        raise ValueError("Loaded streaming schema sample is empty.")

    label_column = resolve_label_column(sample_dataframe, config, args.label_column)
    feature_spec = infer_feature_spec(sample_dataframe, config, label_column)

    training_config = config.get("training", {})
    seed = int(training_config.get("seed", 42))
    validation_fraction = float(training_config.get("validation_fraction", 0.2))
    parquet_batch_rows = int(data_config.get("streaming_batch_rows", 100_000))
    if parquet_batch_rows <= 0:
        raise ValueError("data.streaming_batch_rows must be positive.")
    row_limit = int(data_config["row_limit"]) if data_config.get("row_limit") else None

    stats: dict[str, Any] = {
        "rows": 0,
        "train_rows": 0,
        "validation_rows": 0,
        "dropped_rows": 0,
        "train_positives": 0.0,
        "numeric_sum": np.zeros(len(feature_spec.numeric_columns), dtype="float64"),
        "numeric_sum_squares": np.zeros(len(feature_spec.numeric_columns), dtype="float64"),
    }
    writers: dict[str, Any] = {}

    print(f"Preprocessing streaming Parquet from: {source_path}", flush=True)
    print(f"Writing preprocessed dataset to: {output_dir}", flush=True)
    print(f"Parquet batch rows: {parquet_batch_rows}", flush=True)
    if row_limit:
        print(f"Row limit: {row_limit}", flush=True)
    print_feature_summary(feature_spec)

    try:
        for batch_index, (dataframe, row_indices) in enumerate(
            parquet_batches_to_dataframes(
                source_path,
                feature_columns(feature_spec),
                parquet_batch_rows=parquet_batch_rows,
                row_limit=row_limit,
            ),
            start=1,
        ):
            stats["rows"] += len(dataframe)
            if validation_fraction > 0.0:
                validation_mask = validation_mask_for_indices(
                    row_indices,
                    validation_fraction=validation_fraction,
                    seed=seed,
                )
            else:
                validation_mask = np.zeros(len(dataframe), dtype=bool)

            for split_name, mask in (
                ("train", ~validation_mask),
                ("validation", validation_mask),
            ):
                if not mask.any():
                    continue
                prepared, dropped_rows = prepare_streaming_dataframe(
                    dataframe.loc[mask].reset_index(drop=True),
                    feature_spec,
                    config,
                )
                stats["dropped_rows"] += dropped_rows
                if prepared.empty:
                    continue

                row_count = len(prepared)
                if split_name == "train":
                    stats["train_rows"] += row_count
                    labels = prepared[feature_spec.label_column]
                    stats["train_positives"] += float((labels >= 0.5).sum())
                    if feature_spec.numeric_columns:
                        numeric_values = prepared.loc[:, feature_spec.numeric_columns].to_numpy(dtype="float64")
                        stats["numeric_sum"] += numeric_values.sum(axis=0)
                        stats["numeric_sum_squares"] += np.square(numeric_values).sum(axis=0)
                else:
                    stats["validation_rows"] += row_count

                write_preprocessed_frame(
                    prepared,
                    split_name=split_name,
                    output_dir=output_dir,
                    writers=writers,
                )

            print(
                "Processed batch "
                f"{batch_index}: source rows={stats['rows']}, "
                f"train rows={stats['train_rows']}, "
                f"validation rows={stats['validation_rows']}, "
                f"dropped rows={stats['dropped_rows']}",
                flush=True,
            )
    finally:
        close_parquet_writers(writers)

    if stats["train_rows"] <= 0:
        raise ValueError("No training rows remain after preprocessing.")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source_path),
        "feature_spec": feature_spec.to_dict(),
        "stats": serializable_preprocessing_stats(stats),
        "validation_fraction": validation_fraction,
        "seed": seed,
        "parquet_batch_rows": parquet_batch_rows,
        "row_limit": row_limit,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prepared_config = build_preprocessed_training_config(
        config,
        output_dir=output_dir,
        feature_spec=feature_spec,
    )
    training_config_path = output_dir / "training_config.yaml"
    training_config_path.write_text(
        yaml.safe_dump(prepared_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print("Preprocessed dataset written:", flush=True)
    print(f"  train: {output_dir / 'train.parquet'}", flush=True)
    if stats["validation_rows"]:
        print(f"  validation: {output_dir / 'validation.parquet'}", flush=True)
    print(f"  metadata: {metadata_path}", flush=True)
    print(f"  training_config: {training_config_path}", flush=True)
    print(
        "Train with: "
        f"{Path('.venv/bin/python')} main.py --config {training_config_path}",
        flush=True,
    )


def build_training_callbacks(
    training_config: dict[str, Any],
    output_dir: Path,
    *,
    monitor: str,
    skip_export: bool,
) -> list[tf.keras.callbacks.Callback]:
    callbacks: list[tf.keras.callbacks.Callback] = []
    tensorboard_callback = build_tensorboard_callback(
        training_config.setdefault("tensorboard", {}),
        output_dir,
    )
    if tensorboard_callback is not None:
        callbacks.append(tensorboard_callback)
    if not skip_export:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
        ))
    patience = int(training_config.get("early_stopping_patience", 2))
    if patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode="max",
                patience=patience,
                restore_best_weights=True,
            )
        )
    return callbacks


def run_streaming_training(
    config: dict[str, Any],
    *,
    config_path: Path,
    args: argparse.Namespace,
) -> None:
    data_config = config.get("data", {})
    path = resolve_local_data_path(data_config, config_path=config_path)
    sample_dataframe = streaming_schema_sample(path, data_config)
    if sample_dataframe.empty:
        raise ValueError("Loaded streaming schema sample is empty.")

    label_column = resolve_label_column(sample_dataframe, config, args.label_column)
    feature_spec = infer_feature_spec(sample_dataframe, config, label_column)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    features_config = config.get("features", {})

    seed = int(training_config.get("seed", 42))
    validation_fraction = float(training_config.get("validation_fraction", 0.2))
    batch_size = int(training_config.get("batch_size", 1024))
    parquet_batch_rows = int(data_config.get("streaming_batch_rows", 100_000))
    if parquet_batch_rows <= 0:
        raise ValueError("data.streaming_batch_rows must be positive.")
    row_limit = int(data_config["row_limit"]) if data_config.get("row_limit") else None

    print(f"Streaming Parquet training from: {path}")
    print(f"Parquet batch rows: {parquet_batch_rows}")
    if row_limit:
        print(f"Row limit: {row_limit}")

    stats = collect_streaming_training_stats(
        path,
        feature_spec,
        config,
        parquet_batch_rows=parquet_batch_rows,
        row_limit=row_limit,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    print_feature_summary(feature_spec, train_size=int(stats["rows"]))
    if stats["dropped_rows"]:
        print(f"Dropped rows with missing labels: {stats['dropped_rows']}")
    if stats["validation_rows"]:
        print(f"Train rows: {stats['train_rows']}, validation rows: {stats['validation_rows']}")
    else:
        print(f"Train rows: {stats['train_rows']}, validation disabled")

    numeric_normalizer = build_numeric_normalizer_from_stats(
        feature_spec.numeric_columns,
        stats["numeric_sum"],
        stats["numeric_sum_squares"],
        int(stats["train_rows"]),
    )
    model = build_homepage_recommendation_model(
        feature_spec,
        hidden_units=tuple(int(value) for value in model_config.get("hidden_units", [256, 128, 64])),
        dropout=float(model_config.get("dropout", 0.2)),
        learning_rate=float(training_config.get("learning_rate", 0.001)),
        categorical_hash_bins=int(features_config.get("categorical_hash_bins", 100_000)),
        embedding_dim=int(features_config.get("embedding_dim", 16)),
        activation=str(model_config.get("activation", "relu")),
        l2=float(model_config.get("l2", 0.0)),
        numeric_normalizer=numeric_normalizer,
    )
    model.summary()

    class_weight = build_balanced_class_weight_from_counts(
        float(stats["train_positives"]),
        float(stats["train_rows"]),
        training_config.get("class_weight"),
    )

    train_dataset = build_streaming_dataset(
        path,
        feature_spec,
        config,
        parquet_batch_rows=parquet_batch_rows,
        row_limit=row_limit,
        validation_fraction=validation_fraction,
        seed=seed,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        class_weight=class_weight,
    )
    validation_dataset = None
    validation_steps = None
    if int(stats["validation_rows"]) > 0:
        validation_dataset = build_streaming_dataset(
            path,
            feature_spec,
            config,
            parquet_batch_rows=parquet_batch_rows,
            row_limit=row_limit,
            validation_fraction=validation_fraction,
            seed=seed,
            split="validation",
            batch_size=batch_size,
            shuffle=False,
        )
        validation_steps = math.ceil(int(stats["validation_rows"]) / batch_size)

    output_dir = Path(training_config.get("output_dir", "models/homepage_dnn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = "val_auc" if validation_dataset is not None else "auc"
    callbacks = build_training_callbacks(
        training_config,
        output_dir,
        monitor=monitor,
        skip_export=args.skip_export,
    )

    steps_per_epoch = math.ceil(int(stats["train_rows"]) / batch_size)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(training_config.get("epochs", 10)),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    evaluation_dataset = validation_dataset or train_dataset
    evaluation_steps = validation_steps or steps_per_epoch
    evaluation = model.evaluate(
        evaluation_dataset,
        steps=evaluation_steps,
        return_dict=True,
    )
    print("Evaluation:")
    print(json.dumps({metric: float(value) for metric, value in evaluation.items()}, indent=2))

    if args.skip_export:
        print("Skipping model export because --skip-export was set.")
        return

    export_model(
        model,
        output_dir,
        feature_spec=feature_spec,
        history=history,
        evaluation=evaluation,
        config=config,
    )


def run_preprocessed_training(
    config: dict[str, Any],
    *,
    config_path: Path,
    args: argparse.Namespace,
) -> None:
    data_config = config.get("data", {})
    data_dir = resolve_preprocessed_data_dir(data_config, config_path=config_path)
    metadata = load_preprocessed_metadata(data_dir)
    feature_spec = feature_spec_from_dict(metadata["feature_spec"])

    train_path = data_dir / "train.parquet"
    validation_path = data_dir / "validation.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Preprocessed train split not found: {train_path}")
    if not validation_path.exists():
        validation_path = None

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    features_config = config.get("features", {})

    seed = int(training_config.get("seed", metadata.get("seed", 42)))
    batch_size = int(training_config.get("batch_size", 1024))
    parquet_batch_rows = int(data_config.get("streaming_batch_rows", metadata.get("parquet_batch_rows", 100_000)))
    if parquet_batch_rows <= 0:
        raise ValueError("data.streaming_batch_rows must be positive.")
    row_limit = int(data_config["row_limit"]) if data_config.get("row_limit") else None

    if row_limit is None:
        stats = preprocessing_stats_from_metadata(metadata)
    else:
        stats = collect_preprocessed_training_stats(
            train_path,
            validation_path,
            feature_spec,
            parquet_batch_rows=parquet_batch_rows,
            row_limit=row_limit,
        )

    print(f"Training from preprocessed dataset: {data_dir}", flush=True)
    print(f"Parquet batch rows: {parquet_batch_rows}", flush=True)
    if row_limit:
        print(f"Row limit per split: {row_limit}", flush=True)

    print_feature_summary(feature_spec, train_size=int(stats["rows"]))
    if int(stats["validation_rows"]):
        print(f"Train rows: {stats['train_rows']}, validation rows: {stats['validation_rows']}")
    else:
        print(f"Train rows: {stats['train_rows']}, validation disabled")

    numeric_normalizer = build_numeric_normalizer_from_stats(
        feature_spec.numeric_columns,
        stats["numeric_sum"],
        stats["numeric_sum_squares"],
        int(stats["train_rows"]),
    )
    model = build_homepage_recommendation_model(
        feature_spec,
        hidden_units=tuple(int(value) for value in model_config.get("hidden_units", [256, 128, 64])),
        dropout=float(model_config.get("dropout", 0.2)),
        learning_rate=float(training_config.get("learning_rate", 0.001)),
        categorical_hash_bins=int(features_config.get("categorical_hash_bins", 100_000)),
        embedding_dim=int(features_config.get("embedding_dim", 16)),
        activation=str(model_config.get("activation", "relu")),
        l2=float(model_config.get("l2", 0.0)),
        numeric_normalizer=numeric_normalizer,
    )
    model.summary()

    class_weight = build_balanced_class_weight_from_counts(
        float(stats["train_positives"]),
        float(stats["train_rows"]),
        training_config.get("class_weight"),
    )

    train_dataset = build_preprocessed_dataset(
        train_path,
        feature_spec,
        parquet_batch_rows=parquet_batch_rows,
        row_limit=row_limit,
        seed=seed,
        batch_size=batch_size,
        shuffle=True,
        class_weight=class_weight,
    )
    validation_dataset = None
    validation_steps = None
    if validation_path is not None and int(stats["validation_rows"]) > 0:
        validation_dataset = build_preprocessed_dataset(
            validation_path,
            feature_spec,
            parquet_batch_rows=parquet_batch_rows,
            row_limit=row_limit,
            seed=seed,
            batch_size=batch_size,
            shuffle=False,
        )
        validation_steps = math.ceil(int(stats["validation_rows"]) / batch_size)

    output_dir = Path(training_config.get("output_dir", "models/homepage_dnn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = "val_auc" if validation_dataset is not None else "auc"
    callbacks = build_training_callbacks(
        training_config,
        output_dir,
        monitor=monitor,
        skip_export=args.skip_export,
    )

    steps_per_epoch = math.ceil(int(stats["train_rows"]) / batch_size)
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(training_config.get("epochs", 10)),
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    evaluation_dataset = validation_dataset or train_dataset
    evaluation_steps = validation_steps or steps_per_epoch
    evaluation = model.evaluate(
        evaluation_dataset,
        steps=evaluation_steps,
        return_dict=True,
    )
    print("Evaluation:")
    print(json.dumps({metric: float(value) for metric, value in evaluation.items()}, indent=2))

    if args.skip_export:
        print("Skipping model export because --skip-export was set.")
        return

    export_model(
        model,
        output_dir,
        feature_spec=feature_spec,
        history=history,
        evaluation=evaluation,
        config=config,
    )


def run_inspection(dataframe: pd.DataFrame, config: dict[str, Any], args: argparse.Namespace) -> None:
    print_schema(dataframe)
    try:
        label_column = resolve_label_column(dataframe, config, args.label_column)
        feature_spec = infer_feature_spec(dataframe, config, label_column)
    except ValueError as exc:
        print(f"Feature inference skipped: {exc}")
        return
    print_feature_summary(feature_spec, train_size=len(dataframe))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    apply_cli_overrides(config, args)

    if args.preprocess_output:
        run_streaming_preprocessing(config, config_path=config_path, args=args)
        return

    if not args.inspect and is_preprocessed_parquet_training(config):
        run_preprocessed_training(config, config_path=config_path, args=args)
        return

    if not args.inspect and is_streaming_parquet_training(config, config_path=config_path):
        run_streaming_training(config, config_path=config_path, args=args)
        return

    dataframe = load_dataframe(config, config_path=config_path)
    if dataframe.empty:
        raise ValueError("Loaded dataframe is empty.")

    if args.inspect:
        run_inspection(dataframe, config, args)
        return

    label_column = resolve_label_column(dataframe, config, args.label_column)
    feature_spec = infer_feature_spec(dataframe, config, label_column)
    prepared, dropped_rows = prepare_dataframe(dataframe, feature_spec, config)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    features_config = config.get("features", {})

    seed = int(training_config.get("seed", 42))
    validation_fraction = float(training_config.get("validation_fraction", 0.2))
    batch_size = int(training_config.get("batch_size", 1024))
    train_df, validation_df = split_dataframe(
        prepared,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    print_feature_summary(feature_spec, train_size=len(prepared))
    if dropped_rows:
        print(f"Dropped rows with missing labels: {dropped_rows}")
    if validation_df is not None:
        print(f"Train rows: {len(train_df)}, validation rows: {len(validation_df)}")
    else:
        print(f"Train rows: {len(train_df)}, validation disabled")

    numeric_normalizer = build_numeric_normalizer(train_df, feature_spec.numeric_columns)
    model = build_homepage_recommendation_model(
        feature_spec,
        hidden_units=tuple(int(value) for value in model_config.get("hidden_units", [256, 128, 64])),
        dropout=float(model_config.get("dropout", 0.2)),
        learning_rate=float(training_config.get("learning_rate", 0.001)),
        categorical_hash_bins=int(features_config.get("categorical_hash_bins", 100_000)),
        embedding_dim=int(features_config.get("embedding_dim", 16)),
        activation=str(model_config.get("activation", "relu")),
        l2=float(model_config.get("l2", 0.0)),
        numeric_normalizer=numeric_normalizer,
    )
    model.summary()

    train_dataset = dataframe_to_dataset(
        train_df,
        feature_spec,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_dataset = (
        dataframe_to_dataset(
            validation_df,
            feature_spec,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        if validation_df is not None
        else None
    )

    output_dir = Path(training_config.get("output_dir", "models/homepage_dnn"))
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = "val_auc" if validation_dataset is not None else "auc"
    callbacks = build_training_callbacks(
        training_config,
        output_dir,
        monitor=monitor,
        skip_export=args.skip_export,
    )

    class_weight = build_class_weight(
        train_df[feature_spec.label_column],
        training_config.get("class_weight"),
    )

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(training_config.get("epochs", 10)),
        callbacks=callbacks,
        class_weight=class_weight,
    )

    evaluation_dataset = validation_dataset or train_dataset
    evaluation = model.evaluate(evaluation_dataset, return_dict=True)
    print("Evaluation:")
    print(json.dumps({metric: float(value) for metric, value in evaluation.items()}, indent=2))

    if args.skip_export:
        print("Skipping model export because --skip-export was set.")
        return

    export_model(
        model,
        output_dir,
        feature_spec=feature_spec,
        history=history,
        evaluation=evaluation,
        config=config,
    )


if __name__ == "__main__":
    main()
