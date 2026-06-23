"""Deep Interest Network model."""

import torch
from torch import nn


class DIN(nn.Module):
    """DIN-style attention model for ranking."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        candidate_item_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        candidate_embedding = self.embedding(candidate_item_ids)
        history_embeddings = self.embedding(history_item_ids)

        candidate_expanded = candidate_embedding.unsqueeze(1).expand_as(history_embeddings)
        attention_input = torch.cat(
            [
                candidate_expanded,
                history_embeddings,
                candidate_expanded - history_embeddings,
                candidate_expanded * history_embeddings,
            ],
            dim=-1,
        )
        attention_scores = self.attention(attention_input).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=-1).unsqueeze(-1)
        interest_embedding = torch.sum(attention_weights * history_embeddings, dim=1)

        features = torch.cat([candidate_embedding, interest_embedding], dim=-1)
        return self.mlp(features).squeeze(-1)
