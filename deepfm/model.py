"""DeepFM model."""

import torch
from torch import nn


class DeepFM(nn.Module):
    """DeepFM model for CTR prediction."""

    def __init__(
        self,
        num_fields: int,
        vocab_size: int,
        embedding_dim: int,
        mlp_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_fields = num_fields
        self.linear_embedding = nn.Embedding(vocab_size, 1)
        self.feature_embedding = nn.Embedding(vocab_size, embedding_dim)

        layers: list[nn.Module] = []
        input_dim = num_fields * embedding_dim
        for hidden_dim in mlp_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feature_ids: torch.Tensor) -> torch.Tensor:
        linear_part = self.linear_embedding(feature_ids).sum(dim=1)

        embeddings = self.feature_embedding(feature_ids)
        square_of_sum = torch.sum(embeddings, dim=1) ** 2
        sum_of_square = torch.sum(embeddings**2, dim=1)
        fm_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        deep_part = self.mlp(embeddings.flatten(start_dim=1))
        return (linear_part + fm_part + deep_part).squeeze(-1)
