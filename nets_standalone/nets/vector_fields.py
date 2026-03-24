from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, vmap

from .nn import FeedForward, GaussianFourierEncoder


def cuda_profile(fn):
    @wraps(fn)
    def wrapper(*args, profile: bool = False, **kwargs):
        if profile and torch.cuda.is_available():
            start_bytes = torch.cuda.memory_allocated()
            result = fn(*args, **kwargs)
            end_bytes = torch.cuda.memory_allocated()
            gib = (end_bytes - start_bytes) / (1024 * 1024 * 1024)
            print(f"Call to {fn.__name__} used {gib:.3f} GiB of memory")
            return result
        return fn(*args, **kwargs)

    return wrapper


class VectorField(nn.Module, ABC):
    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.drift(x, t)

    def autograd_divergence(self, x: Tensor, t: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        batched_jacs = vmap(jacrev(self, argnums=0))(x, t).squeeze(1, 3)
        return torch.einsum("bii->b", batched_jacs).unsqueeze(-1)

    @cuda_profile
    def divergence(self, x: Tensor, t: Tensor, mode: str) -> Tensor:
        if mode != "autograd":
            raise NotImplementedError("Standalone repo only includes autograd divergence.")
        return self.autograd_divergence(x, t)


class MLPVectorField(VectorField):
    def __init__(
        self,
        data_dim: int,
        hidden_dims: list[int],
        use_fourier: bool,
        x_fourier_dim: int,
        x_fourier_sigma: float,
        t_fourier_dim: int,
        t_fourier_sigma: float,
    ):
        super().__init__()
        if use_fourier:
            input_dim = x_fourier_dim + t_fourier_dim
            self.mlp = FeedForward([input_dim] + hidden_dims + [data_dim])
            self.x_encoder = GaussianFourierEncoder(data_dim, x_fourier_dim, x_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)

            def net(x: Tensor, t: Tensor) -> Tensor:
                return self.mlp(torch.cat([self.x_encoder(x), self.t_encoder(t)], dim=-1))

        else:
            self.mlp = FeedForward([data_dim + 1] + hidden_dims + [data_dim])

            def net(x: Tensor, t: Tensor) -> Tensor:
                return self.mlp(torch.cat([x, t], dim=-1))

        self.net = net

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(x, t)
