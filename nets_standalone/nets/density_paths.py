from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, vmap

from .base import Density, Sampleable
from .config import Config
from .distributions import GMM, Gaussian
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


class DensityPath(nn.Module, ABC):
    @property
    def dim(self) -> int:
        return self.start_sampleable.dim

    @property
    @abstractmethod
    def start_sampleable(self) -> Sampleable:
        raise NotImplementedError

    @property
    def end_sampleable(self) -> Sampleable:
        raise NotImplementedError

    @abstractmethod
    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    @cuda_profile
    def dt_log_density(self, x: Tensor, t: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        dt_log_density = vmap(jacrev(self.log_density, argnums=1))(x, t)
        return dt_log_density.squeeze((2, 3, 4))

    @cuda_profile
    def dx_log_density(self, x: Tensor, t: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        dx_log_density = vmap(jacrev(self.log_density, argnums=0))(x, t)
        return dx_log_density.squeeze((1, 2, 3))


class LinearDensityPath(DensityPath):
    def __init__(
        self,
        start_sampleable: Sampleable,
        start: Density,
        end: Density,
        end_sampleable: Sampleable,
    ):
        super().__init__()
        self.register_module("_start_sampleable", start_sampleable)
        self.register_module("start", start)
        self.register_module("end", end)
        self.register_module("_end_sampleable", end_sampleable)

    @property
    def start_sampleable(self) -> Sampleable:
        return self._start_sampleable

    @property
    def end_sampleable(self) -> Sampleable:
        return self._end_sampleable

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        return t * self.end.log_density(x) + (1 - t) * self.start.log_density(x)


class LearnableLinearDensityPath(LinearDensityPath):
    def __init__(
        self,
        start_sampleable: Sampleable,
        start: Density,
        end: Density,
        learnable_hiddens: list[int],
        use_fourier: bool,
        x_fourier_dim: int,
        x_fourier_sigma: float,
        t_fourier_dim: int,
        t_fourier_sigma: float,
        end_sampleable: Sampleable,
    ):
        super().__init__(start_sampleable, start, end, end_sampleable)
        if use_fourier:
            input_dim = x_fourier_dim + t_fourier_dim
            self.mlp = FeedForward([input_dim] + learnable_hiddens + [1])
            self.x_encoder = GaussianFourierEncoder(self.dim, x_fourier_dim, x_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)

            def learned_potential(x: Tensor, t: Tensor) -> Tensor:
                return self.mlp(torch.cat([self.x_encoder(x), self.t_encoder(t)], dim=-1))

        else:
            self.mlp = FeedForward([self.dim + 1] + learnable_hiddens + [1])

            def learned_potential(x: Tensor, t: Tensor) -> Tensor:
                return self.mlp(torch.cat([x, t], dim=-1))

        self.learned_potential = learned_potential

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        geometric_component = super().log_density(x, t)
        learned_component = t * (1 - t) * self.learned_potential(x, t)
        return geometric_component + learned_component


def build_density_path(cfg: Config) -> DensityPath:
    if cfg.density_path != "learnable_linear":
        raise NotImplementedError("Standalone repo only includes learnable_linear density path.")

    if cfg.target == "fab_gmm":
        target = GMM.FAB_GMM(cov_scale=cfg.cov_scale)
    elif cfg.target == "symmetric_gmm_2d":
        target = GMM.symmetric_2d(
            nmodes=int(cfg.target_nmodes),
            scale=float(cfg.target_scale),
            std=float(cfg.target_std),
        )
    else:
        raise NotImplementedError("Standalone repo only includes fab_gmm and symmetric_gmm_2d targets.")

    source = Gaussian.isotropic(dim=cfg.x_dim, std=cfg.source_std)
    return LearnableLinearDensityPath(
        start_sampleable=source,
        start=source,
        end=target,
        learnable_hiddens=list(cfg.learnable_hiddens),
        use_fourier=bool(cfg.use_fourier),
        x_fourier_dim=int(cfg.x_fourier_dim),
        x_fourier_sigma=float(cfg.x_fourier_sigma),
        t_fourier_dim=int(cfg.t_fourier_dim),
        t_fourier_sigma=float(cfg.t_fourier_sigma),
        end_sampleable=target,
    )
