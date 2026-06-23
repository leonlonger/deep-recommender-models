# Deep Recommender Models

This repository stores deep learning recommendation model implementations.
It is organized so multiple models can share common embedding, training, and
evaluation utilities while keeping each model implementation isolated.

## Structure

```text
.
├── configs/                    # Model and experiment configs
├── scripts/                    # Training/evaluation entry points
├── src/deep_recommender_models/
│   ├── models/                 # Model implementations
│   └── typing.py               # Shared type aliases
└── tests/                      # Unit tests
```

## Included Models

- `MatrixFactorization`: latent factor baseline for user-item interactions.
- `WideAndDeep`: simple wide-and-deep ranking model skeleton.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Adding A Model

1. Create a file under `src/deep_recommender_models/models/`.
2. Subclass `BaseRecommender`.
3. Export the model in `src/deep_recommender_models/models/__init__.py`.
4. Add focused tests in `tests/`.
