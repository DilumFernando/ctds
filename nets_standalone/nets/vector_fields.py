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
    def __init__(self):
        super().__init__()

    @abstractmethod
    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            drift: (batch_size, dim)
        """
        pass

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            drift: (batch_size, dim)
        """
        return self.drift(x, t)

    def autograd_divergence(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            div: (batch_size, 1)
        """
        # Create a pseudo-batch dimension
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        t = t.unsqueeze(1)  # (batch_size, 1, 1)
        
        # Compute the Jacobian of the vector field with respect to x
        batched_jacs = vmap(jacrev(self, argnums=0))(
            x, t
        )  # (batch_size, 1, dim, 1, dim)
        batched_jacs = batched_jacs.squeeze(1, 3)  # (batch_size, dim, dim)
        # (batch_size, 1)
        return torch.einsum("bii->b", batched_jacs).unsqueeze(-1)

    def hutch_numeric_divergence(
        self, x: Tensor, t: Tensor, delta: float = 0.01
    ) -> Tensor:
        """
        Unbiased estimate of divergence (when delta -> 0). Growing linearly with dim.
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
            delta: used for numerical approximation of the Jacobian-vector product.
        Returns:
            div: (batch_size, 1)
        """
        noise = torch.randn_like(x)  # (bs, dim)
        x_fwd = x + delta * noise
        x_bwd = x - delta * noise
        v_fwd = self(x_fwd, t)  # (bs, dim)
        v_bwd = self(x_bwd, t)  # (bs, dim)
        diff = (v_fwd - v_bwd) / (2 * delta)  # (bs, dim)
        return torch.einsum("bi,bi->b", noise, diff).unsqueeze(-1)

    def hutch_jvp_divergence(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Hutchinson trace estimator; grows linearly with dim.
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            div: (batch_size, 1)
        """
        noise = torch.rand_like(x)  # (bs, dim)

        def x_only(x: Tensor) -> Tensor:
            return self(x, t)

        _, cjvp = jvp(x_only, (x,), (noise,))  # (bs, dim)
        return torch.einsum("bi,bi->b", noise, cjvp).unsqueeze(-1)  # (bs, 1)

    @cuda_profile
    def divergence(self, x: Tensor, t: Tensor, mode: str) -> Tensor:
        """
        Returns divergence with respect to x
        Args:
            xt: (batch_size, dim)
            t: (batch_size, 1)
            mode: "hutch_numeric" | "hutch_jvp" | "autograd"
        Returns:
            div: (batch_size, 1)
        """
        if mode == "hutch_numeric":
            return self.hutch_numeric_divergence(x, t)
        elif mode == "hutch_jvp":
            return self.hutch_jvp_divergence(x, t)
        elif mode == "autograd":
            return self.autograd_divergence(x, t)
    
class MLPVectorField(VectorField):
    """
    MLP-parameterized vector field R^{data_dim + time_embed_dim} -> R^{data_dim}
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dims: List[int],
        use_fourier: bool,
        x_fourier_dim: Optional[int] = None,
        x_fourier_sigma: Optional[float] = None,
        t_fourier_dim: Optional[int] = None,
        t_fourier_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.use_fourier = use_fourier
        if self.use_fourier:
            assert x_fourier_dim is not None and t_fourier_dim is not None
            assert x_fourier_sigma is not None and t_fourier_sigma is not None
            input_dim = x_fourier_dim + t_fourier_dim
            self.mlp = FeedForward([input_dim] + hidden_dims + [data_dim])
            self.x_encoder = GaussianFourierEncoder(data_dim, x_fourier_dim, x_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)

            def fwd(x: Tensor, t: Tensor) -> Tensor:
                x_encoding = self.x_encoder(x)
                t_encoding = self.t_encoder(t)
                xt_encoding = torch.cat([x_encoding, t_encoding], dim=-1)
                return self.mlp(xt_encoding)

            self.net = fwd
        else:
            self.mlp = FeedForward([data_dim + 1] + hidden_dims + [data_dim])

            def fwd(x: Tensor, t: Tensor) -> Tensor:
                x_t = torch.cat([x, t], dim=-1)
                return self.mlp(x_t)

            self.net = fwd

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(x, t)