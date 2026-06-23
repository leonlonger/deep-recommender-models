"""Base abstractions for recommender models."""

from abc import ABC, abstractmethod

from torch import nn

from deep_recommender_models.typing import FeatureBatch, Tensor


class BaseRecommender(nn.Module, ABC):
    """Common interface for recommendation models."""

    @abstractmethod
    def forward(self, features: FeatureBatch) -> Tensor:
        """Return logits or scores for a batch of examples."""
