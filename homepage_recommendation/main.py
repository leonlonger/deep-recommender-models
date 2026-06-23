"""Homepage recommendation training entry."""

import torch

from model import HomepageRecommendationModel


def main() -> None:
    model = HomepageRecommendationModel(
        user_feature_dim=64,
        item_feature_dim=64,
        hidden_dims=(128, 64),
        dropout=0.1,
    )

    user_features = torch.randn(4, 64)
    item_features = torch.randn(4, 64)
    scores = model(user_features, item_features)
    print(scores)


if __name__ == "__main__":
    main()
