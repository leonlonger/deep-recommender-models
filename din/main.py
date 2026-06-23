"""DIN training entry."""

import torch

from model import DIN


def main() -> None:
    model = DIN(vocab_size=100000, embedding_dim=32, hidden_dims=(128, 64), dropout=0.1)

    candidate_item_ids = torch.randint(0, 100000, (4,))
    history_item_ids = torch.randint(0, 100000, (4, 20))
    logits = model(candidate_item_ids, history_item_ids)
    print(logits)


if __name__ == "__main__":
    main()
