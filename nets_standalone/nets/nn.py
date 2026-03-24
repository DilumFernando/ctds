from __future__ import annotations

import math
from typing import List, Type

import torch
import torch.nn as nn
from torch import Tensor


class GaussianFourierEncoder(nn.Module):
    def __init__(self, input_dim: int, fourier_dim: int, sigma: float):
        super().__init__()
        assert fourier_dim % 2 == 0
        half_dim = fourier_dim // 2
        self.register_buffer("B", torch.randn(half_dim, input_dim) * sigma)

    def forward(self, x: Tensor) -> Tensor:
        x_proj = 2 * math.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, unit_dims: List[int], activation: Type[nn.Module] = nn.SiLU):
        super().__init__()
        layers = []
        for idx in range(len(unit_dims) - 1):
            layers.append(nn.Linear(unit_dims[idx], unit_dims[idx + 1]))
            if idx < len(unit_dims) - 2:
                layers.append(activation())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)
