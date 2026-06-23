"""Wide-and-deep ranking model."""

import torch
from torch import nn

from deep_recommender_models.models.base import BaseRecommender
from deep_recommender_models.typing import FeatureBatch, Tensor


class WideAndDeep(BaseRecommender):
    """Combine memorization features with dense neural ranking features."""

    def __init__(
        self,
        wide_dim: int,
        dense_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.wide = nn.Linear(wide_dim, 1)

        layers: list[nn.Module] = []
        input_dim = dense_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, features: FeatureBatch) -> Tensor:
        wide_features = features["wide_features"].float()
        dense_features = features["dense_features"].float()

        wide_score = self.wide(wide_features)
        deep_score = self.deep(dense_features)
        return torch.flatten(wide_score + deep_score)
