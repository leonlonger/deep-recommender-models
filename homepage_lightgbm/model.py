"""LightGBM utilities for homepage CTR ranking."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """Scalar feature columns consumed by the LightGBM CTR model."""

    label_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureSpec":
        return cls(
            label_column=str(value["label_column"]),
            numeric_columns=tuple(str(column) for column in value.get("numeric_columns", [])),
            categorical_columns=tuple(str(column) for column in value.get("categorical_columns", [])),
        )

    def with_label(self, label_column: str) -> "FeatureSpec":
        return FeatureSpec(
            label_column=label_column,
            numeric_columns=self.numeric_columns,
            categorical_columns=self.categorical_columns,
        )

    @property
    def feature_columns(self) -> list[str]:
        return [*self.numeric_columns, *self.categorical_columns]

    @property
    def required_columns(self) -> list[str]:
        return [self.label_column, *self.feature_columns]


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


def categorical_series_to_strings(series: pd.Series) -> pd.Series:
    return series.astype("object").map(categorical_value_to_string)


@dataclass
class FeatureEncoder:
    """Deterministic categorical encoder used before LightGBM Dataset creation."""

    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    category_maps: dict[str, dict[str, int]]
    unknown_category_code: int = -1

    @classmethod
    def fit(cls, dataframe: pd.DataFrame, feature_spec: FeatureSpec) -> "FeatureEncoder":
        category_maps: dict[str, dict[str, int]] = {}
        for column in feature_spec.categorical_columns:
            values = categorical_series_to_strings(dataframe[column])
            categories = sorted(str(value) for value in values.drop_duplicates())
            category_maps[column] = {category: index for index, category in enumerate(categories)}

        return cls(
            numeric_columns=feature_spec.numeric_columns,
            categorical_columns=feature_spec.categorical_columns,
            category_maps=category_maps,
        )

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        encoded: dict[str, pd.Series] = {}
        for column in self.numeric_columns:
            values = pd.to_numeric(dataframe[column], errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            encoded[column] = values.astype("float32")

        for column in self.categorical_columns:
            mapping = self.category_maps[column]
            values = categorical_series_to_strings(dataframe[column])
            codes = values.map(mapping).fillna(self.unknown_category_code)
            encoded[column] = codes.astype("int32")

        if not encoded:
            raise ValueError("No model features were selected.")
        return pd.DataFrame(encoded, index=dataframe.index)

    @property
    def categorical_feature_names(self) -> list[str]:
        return list(self.categorical_columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "category_maps": self.category_maps,
            "unknown_category_code": self.unknown_category_code,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureEncoder":
        return cls(
            numeric_columns=tuple(str(column) for column in value.get("numeric_columns", [])),
            categorical_columns=tuple(str(column) for column in value.get("categorical_columns", [])),
            category_maps={
                str(column): {str(category): int(code) for category, code in mapping.items()}
                for column, mapping in value.get("category_maps", {}).items()
            },
            unknown_category_code=int(value.get("unknown_category_code", -1)),
        )

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_lgb_dataset(
    features: pd.DataFrame,
    labels: np.ndarray,
    *,
    categorical_feature_names: list[str],
    weights: np.ndarray | None = None,
    reference: Any | None = None,
) -> Any:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Missing dependency lightgbm. Install with: pip install -r homepage_lightgbm/requirements.txt"
        ) from exc

    categorical_feature = categorical_feature_names or "auto"
    return lgb.Dataset(
        features,
        label=labels,
        weight=weights,
        reference=reference,
        feature_name=list(features.columns),
        categorical_feature=categorical_feature,
        free_raw_data=False,
    )


def binary_logloss(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    clipped = np.clip(predictions.astype("float64"), 1e-15, 1.0 - 1e-15)
    labels = labels.astype("float64")
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def roc_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = pd.Series(predictions).rank(method="average").to_numpy(dtype="float64")
    positive_rank_sum = float(ranks[labels.astype(bool)].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def pr_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    positives = float(labels.sum())
    if positives == 0.0:
        return float("nan")
    order = np.argsort(-predictions)
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    false_positives = np.cumsum(1.0 - sorted_labels)
    precision = true_positives / np.maximum(true_positives + false_positives, 1.0)
    recall = true_positives / positives
    precision_points = np.concatenate(([1.0], precision))
    recall_points = np.concatenate(([0.0], recall))
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precision_points, recall_points))
    widths = recall_points[1:] - recall_points[:-1]
    heights = (precision_points[1:] + precision_points[:-1]) / 2.0
    return float(np.sum(widths * heights))


def evaluate_binary_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype="float32")
    predictions = np.asarray(predictions, dtype="float32")
    binary_labels = (labels >= threshold).astype("float32")
    binary_predictions = (predictions >= threshold).astype("float32")

    true_positive = float(((binary_predictions == 1.0) & (binary_labels == 1.0)).sum())
    false_positive = float(((binary_predictions == 1.0) & (binary_labels == 0.0)).sum())
    false_negative = float(((binary_predictions == 0.0) & (binary_labels == 1.0)).sum())
    clicked_clicks = float(binary_labels.sum())
    predicted_clicks = float(predictions.sum())

    accuracy = float((binary_predictions == binary_labels).mean()) if len(labels) else float("nan")
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    pcoc = predicted_clicks / clicked_clicks if clicked_clicks else float("nan")

    return {
        "auc": roc_auc(binary_labels, predictions),
        "binary_logloss": binary_logloss(binary_labels, predictions),
        "pr_auc": pr_auc(binary_labels, predictions),
        "pcoc": float(pcoc),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "predicted_clicks": predicted_clicks,
        "clicked_clicks": clicked_clicks,
    }


def feature_importance_dataframe(booster: Any) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "importance_split": booster.feature_importance(importance_type="split"),
            "importance_gain": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values(["importance_gain", "importance_split"], ascending=False)


def clean_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean_metric_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_metric_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
