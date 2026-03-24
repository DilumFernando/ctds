from __future__ import annotations

from functools import cached_property

import numpy as np
import torch
import torch.distributions as D
from torch import Size, Tensor

from .base import Density, Sampleable
from .constants import FAB_GMM_COVARIANCES, FAB_GMM_MEANS, FAB_NMODES


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
