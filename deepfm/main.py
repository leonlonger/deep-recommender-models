"""DeepFM training entry."""

import torch

from model import DeepFM


def main() -> None:
    model = DeepFM(
        num_fields=16,
        vocab_size=100000,
        embedding_dim=16,
        mlp_dims=(128, 64),
        dropout=0.1,
    )

    feature_ids = torch.randint(0, 100000, (4, 16))
    logits = model(feature_ids)
    print(logits)


if __name__ == "__main__":
    main()
