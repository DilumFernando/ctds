from functools import cached_property

import numpy as np
import torch
import torch.distributions as D
from torch import Size, Tensor

from src.systems.base import Density, Sampleable
from src.systems.constants import FAB_GMM_COVARIANCES, FAB_GMM_MEANS, FAB_NMODES


class TorchDensity(Density):
    """
    Wrapper around a torch.distributions.Distribution object.
    """

    def __init__(self, dist: torch.distributions.Distribution):
        super().__init__()
        self.dist = dist

    def log_density(self, x: Tensor) -> Tensor:
        return self.dist.log_prob(x).view(-1, 1)

    def energy(self, x: Tensor) -> Tensor:
        return -self.dist.log_prob(x)

    def raise_to_temperature(self, beta: float) -> "Density":
        raise NotImplementedError()

    def dim(self) -> int:
        return self.dist.event_shape[0]


class Gaussian(Sampleable, Density):
    def __init__(self, mean, cov):
        super().__init__()
        self.register_buffer("mean", mean)  # (dim, )
        self.register_buffer("cov", cov)  # (dim, dim)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std**2
        return cls(mean, cov)

    @property
    def distribution(self):
        """
        Recreate the underlying distribution object dynamically to avoid device mismatch.
        """
        return D.MultivariateNormal(self.mean, self.cov, validate_args=False)

    @property
    def dim(self) -> int:
        return self.mean.shape[0]

    def sample(self, num_samples) -> Tensor:
        return self.distribution.sample(Size((num_samples,)))

    def log_density(self, x: Tensor):
        return self.distribution.log_prob(x).view(-1, 1)

    def energy(self, x: Tensor):
        return -self.log_density(x)

    @staticmethod
    def random(device) -> "Gaussian":
        mean = torch.rand(2).to(device)
        pre_cov = torch.rand(2, 2).to(device)
        cov = torch.mm(pre_cov, pre_cov.t())
        return Gaussian(mean, cov)

    # Gradient of the density
    def log_density_gradient(self, x: Tensor):
        return self.log_grad_fn(x) * torch.exp(self.get_log_density(x)).view(-1, 1)

    # Gradient of the log density
    def log_grad_fn(self, x: Tensor):
        return -torch.matmul((x.float() - self.mean), torch.inverse(self.cov))

    def raise_to_temperature(self, beta: float) -> "Gaussian":
        return Gaussian(self.mean, self.cov / beta)

    def sample_at_temp(self, beta: Tensor) -> Tensor:
        """
        Batched sampling at arbitrary temperatures.
        Args:
            beta: (bs, 1)
        Returns:
            x: (bs, dim)
        """
        beta = beta.unsqueeze(-1)  # (bs, 1, 1)
        cov = self.cov.unsqueeze(0)  # (1, dim, dim)
        return D.MultivariateNormal(self.mean, cov / beta).sample()

    def log_density_at_temp(self, x: Tensor, beta: Tensor) -> Tensor:
        """
        Batched density evaluation at arbitrary temperatures.
        Args:
            x: (bs, dim)
            beta: (bs, 1)
        Returns:
            density: (bs, 1)
        """
        beta = beta.unsqueeze(-1)  # (bs, 1, 1)
        cov = self.cov.unsqueeze(0)  # (1, dim, dim)
        return (
            D.MultivariateNormal(self.mean, cov / beta, validate_args=False)
            .log_prob(x)
            .view(-1, 1)
        )


class GMMDensity(Density):
    def __init__(
        self,
        means: Tensor,  # nmodes x data_dim
        covs: Tensor,  # nmodes x data_dim x data_dim
        weights: Tensor,  # nmodes
        beta: float = 1.0,  # Inverse temperature
    ):
        super().__init__()
        self.nmodes = means.shape[0]
        self.beta = beta

        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def distribution(self):
        return D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=self.weights, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covs,
                validate_args=False,
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
        """
        Args:
            x: (batch_size, dim)
        Returns:
            log_density: (batch_size,)
        """
        return self.beta * self.distribution.log_prob(x).view(-1, 1)

    def energy(self, x: Tensor) -> Tensor:
        return -self.log_density(x)

    def raise_to_temperature(self, beta: float) -> "GMMDensity":
        return GMMDensity(self.means, self.covs, self.weights, self.beta * beta)


class GMM(GMMDensity, Sampleable):
    def __init__(self, means, covs, weights):
        super().__init__(means, covs, weights)

    def sample(self, num_samples: int) -> Tensor:
        assert self.beta == 1.0, "Sampling at arbitrary temperatures is not implemented."
        return self.distribution.sample(torch.Size((num_samples,)))

    @classmethod
    def random(
        cls,
        nmodes: int,
        scale: float,
        dim: int,
        std: float,
        seed: int,
    ) -> "GMM":
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, dim) - 0.5) * scale
        covs = torch.diag_embed(torch.ones(nmodes, dim)) * std**2
        weights = D.Dirichlet(torch.ones(nmodes)).sample()
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(cls, nmodes: int, scale: float = 10.0, std=1.0) -> "GMM":
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std**2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)

    @classmethod
    def FAB_GMM(cls, cov_scale: float = 1.0) -> "GMM":
        return cls(
            FAB_GMM_MEANS, cov_scale * FAB_GMM_COVARIANCES, torch.ones(FAB_NMODES) / FAB_NMODES
        )


class GaussianGMM(GMM):
    """
    A GMM with all means being zero and covariances equal to the identity matrix
    """

    def __init__(self, x_dim: int, std: float, nmodes: int):
        self.std = std

        means = torch.zeros(nmodes, x_dim)
        covs = torch.diag_embed(torch.ones(nmodes, x_dim) * std**2)
        weights = torch.ones(nmodes) / nmodes
        super().__init__(means, covs, weights)

    def sample_at_temp(self, beta: Tensor) -> Tensor:
        """
        Batched sampling at arbitrary temperatures.
        Args:
            beta: (bs, 1)
        Returns:
            x: (bs, dim)
        """
        beta = beta.unsqueeze(-1)  # (bs, 1, 1)
        covs = self.covs[0].unsqueeze(0)  # (1, dim, dim)
        means = self.means[0].unsqueeze(0)  # (1, dim)
        return D.MultivariateNormal(means, covs / beta).sample()

    def raise_to_temperature(self, beta: float) -> "GMMDensity":
        return GaussianGMM(self.dim, self.std / np.sqrt(beta), self.nmodes)