from typing import List, Type
import math

import torch
import torch.nn as nn
from torch import Tensor

class GaussianFourierEncoder(nn.Module):
    """
    Gaussian Fourier positional encoding. Based on https://arxiv.org/abs/2006.10739
    """
    def __init__(self, input_dim: int, fourier_dim: int, sigma: float):
        super().__init__()
        assert fourier_dim % 2 == 0
        half_dim = fourier_dim // 2
        B = torch.randn(half_dim, input_dim) * sigma
        self.register_buffer('B', B)

    def forward(self, x: Tensor):
        """
        Args:
            s: (bs, input_dim)
        Returns:
            encoding: (bs, fourier_dim)
        """
        x_proj = 2 * math.pi * x @ self.B.T  # (bs, half_dim)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, unit_dims: List[int], activation: Type[nn.Module] = nn.SiLU):
        super().__init__()
        self.unit_dims = unit_dims
        self.activation = activation
        self.mlp = self.build()

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def build(self) -> nn.Module:
        mlp = []
        for idx in range(len(self.unit_dims) - 1):
            mlp.append(nn.Linear(self.unit_dims[idx], self.unit_dims[idx + 1]))
            if idx < len(self.unit_dims) - 2:
                mlp.append(self.activation())
        return nn.Sequential(*mlp)