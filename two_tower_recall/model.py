"""Two-tower recall model."""

import torch
from torch import nn
from torch.nn import functional as F


class TwoTowerRecall(nn.Module):
    """Two-tower retrieval model using dot-product similarity."""

    def __init__(self, user_feature_dim: int, item_feature_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_embedding = F.normalize(self.user_tower(user_features), dim=-1)
        item_embedding = F.normalize(self.item_tower(item_features), dim=-1)
        return torch.sum(user_embedding * item_embedding, dim=-1)
