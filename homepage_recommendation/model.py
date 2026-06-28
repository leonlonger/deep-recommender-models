"""TensorFlow model utilities for homepage recommendation ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass(frozen=True)
class FeatureSpec:
    """Scalar feature columns used by the DNN ranker."""

    label_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@tf.keras.utils.register_keras_serializable(package="homepage_recommendation")
class PCOC(tf.keras.metrics.Metric):
    """Predicted clicks over clicked clicks."""

    def __init__(self, name: str = "pcoc", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.predicted_clicks = self.add_weight(name="predicted_clicks", initializer="zeros")
        self.clicked_clicks = self.add_weight(name="clicked_clicks", initializer="zeros")

    def update_state(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: tf.Tensor | None = None,
    ) -> None:
        y_true = tf.reshape(tf.cast(y_true, self.dtype), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, self.dtype), [-1])

        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, self.dtype), [-1])
            y_true = y_true * sample_weight
            y_pred = y_pred * sample_weight

        self.predicted_clicks.assign_add(tf.reduce_sum(y_pred))
        self.clicked_clicks.assign_add(tf.reduce_sum(y_true))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.predicted_clicks, self.clicked_clicks)

    def reset_state(self) -> None:
        self.predicted_clicks.assign(0.0)
        self.clicked_clicks.assign(0.0)


def build_numeric_normalizer(
    dataframe: pd.DataFrame,
    numeric_columns: tuple[str, ...],
) -> tf.keras.layers.Layer | None:
    """Build and adapt a normalization layer for dense numeric inputs."""

    if not numeric_columns:
        return None

    normalizer = tf.keras.layers.Normalization(
        axis=-1,
        name="numeric_normalization",
    )
    values = dataframe.loc[:, numeric_columns].to_numpy(dtype="float32")
    normalizer.adapt(values)
    return normalizer


def build_homepage_recommendation_model(
    feature_spec: FeatureSpec,
    *,
    hidden_units: tuple[int, ...],
    dropout: float,
    learning_rate: float,
    classification_threshold: float = 0.5,
    categorical_hash_bins: int,
    embedding_dim: int,
    activation: str = "relu",
    l2: float = 0.0,
    numeric_normalizer: tf.keras.layers.Layer | None = None,
) -> tf.keras.Model:
    """Build a simple DNN ranking model for homepage recommendation."""

    inputs: dict[str, tf.keras.layers.Input] = {}
    encoded_features: list[tf.Tensor] = []
    regularizer = tf.keras.regularizers.l2(l2) if l2 else None

    if feature_spec.numeric_columns:
        numeric_inputs = []
        for column in feature_spec.numeric_columns:
            input_layer = tf.keras.Input(shape=(1,), name=column, dtype=tf.float32)
            inputs[column] = input_layer
            numeric_inputs.append(input_layer)

        numeric_tensor = (
            numeric_inputs[0]
            if len(numeric_inputs) == 1
            else tf.keras.layers.Concatenate(name="numeric_features")(numeric_inputs)
        )
        if numeric_normalizer is not None:
            numeric_tensor = numeric_normalizer(numeric_tensor)
        encoded_features.append(numeric_tensor)

    for column in feature_spec.categorical_columns:
        input_layer = tf.keras.Input(shape=(1,), name=column, dtype=tf.string)
        inputs[column] = input_layer
        hashed = tf.keras.layers.Hashing(
            num_bins=categorical_hash_bins,
            name=f"{column}_hash",
        )(input_layer)
        embedded = tf.keras.layers.Embedding(
            input_dim=categorical_hash_bins,
            output_dim=embedding_dim,
            embeddings_regularizer=regularizer,
            name=f"{column}_embedding",
        )(hashed)
        encoded_features.append(
            tf.keras.layers.Flatten(name=f"{column}_embedding_flatten")(embedded)
        )

    if not encoded_features:
        raise ValueError("No model features were selected. Configure numeric or categorical columns.")

    features = (
        encoded_features[0]
        if len(encoded_features) == 1
        else tf.keras.layers.Concatenate(name="all_features")(encoded_features)
    )

    x = features
    for index, units in enumerate(hidden_units, start=1):
        x = tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            name=f"dense_{index}",
        )(x)
        if dropout:
            x = tf.keras.layers.Dropout(dropout, name=f"dropout_{index}")(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid", name="score")(x)
    model = tf.keras.Model(inputs=inputs, outputs=output, name="homepage_dnn_ranker")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(name="binary_crossentropy"),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            PCOC(name="pcoc"),
            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy",
                threshold=classification_threshold,
            ),
            tf.keras.metrics.Precision(
                name="precision",
                thresholds=classification_threshold,
            ),
            tf.keras.metrics.Recall(
                name="recall",
                thresholds=classification_threshold,
            ),
        ],
    )
    return model


def dataframe_to_dataset(
    dataframe: pd.DataFrame,
    feature_spec: FeatureSpec,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Convert a prepared pandas dataframe to a batched tf.data.Dataset."""

    features: dict[str, np.ndarray] = {}
    for column in feature_spec.numeric_columns:
        features[column] = dataframe[column].to_numpy(dtype="float32").reshape(-1, 1)
    for column in feature_spec.categorical_columns:
        features[column] = dataframe[column].astype(str).to_numpy().reshape(-1, 1)

    labels = dataframe[feature_spec.label_column].to_numpy(dtype="float32").reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        buffer_size = min(len(dataframe), max(batch_size * 20, batch_size))
        dataset = dataset.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
