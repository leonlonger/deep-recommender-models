"""Homepage recommendation model."""

import torch
from torch import nn


class HomepageRecommendationModel(nn.Module):
    """Simple ranking model for homepage recommendation candidates."""

    def __init__(
        self,
        user_feature_dim: int,
        item_feature_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = user_feature_dim + item_feature_dim

        layers: list[nn.Module] = []
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
        self.network = nn.Sequential(*layers)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        features = torch.cat([user_features, item_features], dim=-1)
        return self.network(features).squeeze(-1)
