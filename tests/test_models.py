import torch

from deep_recommender_models import MatrixFactorization, WideAndDeep


def test_matrix_factorization_outputs_batch_scores() -> None:
    model = MatrixFactorization(num_users=10, num_items=20, embedding_dim=8)

    scores = model(
        {
            "user_id": torch.tensor([1, 2, 3]),
            "item_id": torch.tensor([4, 5, 6]),
        }
    )

    assert scores.shape == (3,)


def test_wide_and_deep_outputs_batch_scores() -> None:
    model = WideAndDeep(wide_dim=5, dense_dim=7, hidden_dims=(16, 8), dropout=0.0)

    scores = model(
        {
            "wide_features": torch.randn(4, 5),
            "dense_features": torch.randn(4, 7),
        }
    )

    assert scores.shape == (4,)
