"""Shared type aliases."""

from typing import TypeAlias

import torch

Tensor: TypeAlias = torch.Tensor
FeatureBatch: TypeAlias = dict[str, Tensor]
