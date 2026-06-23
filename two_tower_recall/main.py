"""Two-tower recall training entry."""

import torch

from model import TwoTowerRecall


def main() -> None:
    model = TwoTowerRecall(user_feature_dim=128, item_feature_dim=128, embedding_dim=64)

    user_features = torch.randn(4, 128)
    item_features = torch.randn(4, 128)
    scores = model(user_features, item_features)
    print(scores)


if __name__ == "__main__":
    main()
