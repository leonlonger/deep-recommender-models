"""Matrix factorization baseline."""

import torch
from torch import nn

from deep_recommender_models.models.base import BaseRecommender
from deep_recommender_models.typing import FeatureBatch, Tensor


class MatrixFactorization(BaseRecommender):
    """Latent factor model for user-item recommendation."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: FeatureBatch) -> Tensor:
        user_ids = features["user_id"]
        item_ids = features["item_id"]

        user_vectors = self.user_embedding(user_ids)
        item_vectors = self.item_embedding(item_ids)
        interaction = (user_vectors * item_vectors).sum(dim=-1)

        return (
            interaction
            + self.user_bias(user_ids).squeeze(-1)
            + self.item_bias(item_ids).squeeze(-1)
            + self.global_bias
        )
