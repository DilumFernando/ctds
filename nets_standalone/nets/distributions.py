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
    def constrained_random_gmm(
        cls,
        dim: int,
        mode_weights: list[float],
        mode_stds: list[float],
        min_mode_distance: float,
        seed: int,
        max_tries: int = 10_000,
    ) -> "GMM":
        if dim < 1:
            raise ValueError("dim must be at least 1")
        if len(mode_weights) != len(mode_stds):
            raise ValueError("mode_weights and mode_stds must have the same length")
        if len(mode_weights) == 0:
            raise ValueError("At least one mode is required")

        generator = torch.Generator().manual_seed(seed)
        nmodes = len(mode_weights)
        weights = torch.tensor(mode_weights, dtype=torch.float32)
        weights = weights / weights.sum()
        stds = torch.tensor(mode_stds, dtype=torch.float32)

        bounded_min_distance = min(float(min_mode_distance), 2.0 * PLOT_LIMIT)
        means = []
        for _ in range(nmodes):
            accepted = False
            for _attempt in range(max_tries):
                candidate = (torch.rand(dim, generator=generator) * 2.0 - 1.0) * PLOT_LIMIT
                if not means:
                    means.append(candidate)
                    accepted = True
                    break
                stacked = torch.stack(means, dim=0)
                distances = torch.linalg.norm(stacked - candidate.unsqueeze(0), dim=1)
                if torch.all(distances >= bounded_min_distance):
                    means.append(candidate)
                    accepted = True
                    break
            if not accepted:
                raise ValueError(
                    "Could not place all modes within the plotting bounds while satisfying min_mode_distance. "
                    "Try fewer modes or a smaller min_mode_distance."
                )

        means = torch.stack(means, dim=0)
        covs = torch.diag_embed(stds.square().unsqueeze(-1).expand(-1, dim))
        return cls(means, covs, weights)

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

class WarmStartGMMPrior(Density):
    def __init__(
        self,
        means: Tensor,
        beta_max: float,
        beta_decay_rate: float,
        weights: Tensor | None = None,
        fixed_beta: bool = False,
    ):
        super().__init__()
        self.register_buffer("means", means)
        if weights is None:
            weights = torch.ones(means.shape[0], dtype=means.dtype) / means.shape[0]
        weights = weights / weights.sum()
        self.register_buffer("weights", weights)
        self.beta_max = float(beta_max)
        self.beta_decay_rate = float(beta_decay_rate)
        self.fixed_beta = bool(fixed_beta)

    @property
    def dim(self) -> int:
        return self.means.shape[1]

    def beta(self, t: Tensor) -> Tensor:
        if self.fixed_beta:
            return torch.ones_like(t) * self.beta_max
        return self.beta_max * torch.pow(2.0, -self.beta_decay_rate * t)

    def start_sampleable(self) -> GMM:
        variance = 1.0 / self.beta_max
        std = variance**0.5
        covs = torch.diag_embed(torch.ones(self.means.shape[0], self.dim, device=self.means.device) * std**2)
        return GMM(self.means, covs, self.weights)

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        beta_t = self.beta(t)
        variance_t = 1.0 / beta_t
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)
        squared_distance = diff.square().sum(dim=-1)
        log_weights = torch.log(self.weights).unsqueeze(0)
        exponent = -0.5 * squared_distance / variance_t
        return torch.logsumexp(log_weights + exponent, dim=1, keepdim=True)
