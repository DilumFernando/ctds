from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
import torch.distributions as D
from torch import Size, Tensor

from .base import Density, Sampleable
from .constants import FAB_GMM_COVARIANCES, FAB_GMM_MEANS, FAB_NMODES

PLOT_LIMIT = 20.0


class Gaussian(Sampleable, Density):
    def __init__(self, mean: Tensor, cov: Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        return cls(torch.zeros(dim), torch.eye(dim) * std**2)

    @property
    def distribution(self):
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def sample(self, num_samples: int) -> Tensor:
        return self.distribution.sample(Size((num_samples,)))

    def log_density(self, x: Tensor) -> Tensor:
        return self.distribution.log_prob(x).view(-1, 1)


class GMMDensity(Density):
    def __init__(self, means: Tensor, covs: Tensor, weights: Tensor):
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means, covariance_matrix=self.covs, validate_args=False
            ),
            validate_args=False,
        )

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    @cached_property
    def two_dimensional(self) -> bool:
        return self.dim == 2

    def log_density(self, x: Tensor) -> Tensor:
        return self.distribution.log_prob(x).view(-1, 1)


class GMM(GMMDensity, Sampleable):
    def sample(self, num_samples: int) -> Tensor:
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def asymmetric_two_mode(
        cls,
        dim: int,
        mode_distance: float,
        small_mode_weight: float,
        large_mode_weight: float,
        small_mode_std: float,
        large_mode_std: float,
        randomize_mode_locations: bool = False,
        seed: int = 0,
    ) -> "GMM":
        if dim < 1:
            raise ValueError("dim must be at least 1")

        bounded_mode_distance = min(float(mode_distance), 2.0 * PLOT_LIMIT)
        means = torch.zeros(2, dim)
        if randomize_mode_locations and dim > 1:
            generator = torch.Generator().manual_seed(seed)
            direction = torch.randn(dim, generator=generator)
            direction = direction / torch.linalg.norm(direction)
            offset = direction * (bounded_mode_distance / 2.0)
            max_midpoint = torch.full((dim,), PLOT_LIMIT) - torch.abs(offset)
            midpoint = (torch.rand(dim, generator=generator) * 2.0 - 1.0) * max_midpoint
            means[0] = midpoint - offset
            means[1] = midpoint + offset
        else:
            means[0, 0] = -bounded_mode_distance / 2.0
            means[1, 0] = bounded_mode_distance / 2.0

        covs = torch.diag_embed(
            torch.tensor(
                [
                    [large_mode_std**2] * dim,
                    [small_mode_std**2] * dim,
                ],
                dtype=means.dtype,
            )
        )
        weights = torch.tensor([small_mode_weight, large_mode_weight], dtype=means.dtype)
        weights = weights / weights.sum()
        return cls(means, covs, weights)

    @classmethod
    def random_gmm(
        cls,
        nmodes: int,
        scale: float,
        dim: int,
        std: float,
        seed: int,
    ) -> "GMM":
        generator = torch.Generator().manual_seed(seed)
        half_width = min(float(scale) / 2.0, PLOT_LIMIT)
        means = (torch.rand(nmodes, dim, generator=generator) - 0.5) * (2.0 * half_width)
        covs = torch.diag_embed(torch.ones(nmodes, dim) * std**2)
        raw_weights = torch.rand(nmodes, generator=generator)
        weights = raw_weights / raw_weights.sum()
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2d(cls, nmodes: int, scale: float = 10.0, std: float = 1.0) -> "GMM":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std**2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

    @classmethod
    def FAB_GMM(cls, cov_scale: float = 1.0) -> "GMM":
        return cls(
            FAB_GMM_MEANS,
            cov_scale * FAB_GMM_COVARIANCES,
            torch.ones(FAB_NMODES) / FAB_NMODES,
        )
