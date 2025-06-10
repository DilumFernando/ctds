from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from src.systems.base import Density


class Proposal(ABC):
    """Proposal p(x_new | x)"""

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """
        Samples from p(x_new | x) and returns x_new
        Args:
        - x: (bs, dim)
        Returns:
        - x_new: (bs, dim)
        """

        pass

    @abstractmethod
    def log_transition_factor(self, x: Tensor, x_new: Tensor) -> Tensor:
        """
        Computes log p(x_new | x) - log p(x | x_new)
        Args:
        - x: (bs, dim)
        - x_new: (bs, dim)
        Returns:
        - log_transition_factor: (bs,1)
        """
        raise NotImplementedError()


class GaussianProposal(Proposal):
    """p(x_new | x) = N(x_new | x, sigma^2)"""

    def __init__(self, sigma: float):
        self.sigma = sigma

    def __call__(self, x: Tensor) -> Tensor:
        return x + torch.randn_like(x) * self.sigma

    def log_transition_factor(self, x: Tensor, x_new: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], 1).to(x)


class MCMC(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def step(self, x: Tensor) -> Tensor:
        """Perform a single MCMC step which preserves the target distribution."""
        pass


class MetropolisHastings(MCMC):
    def __init__(self, inv_density: Density, proposal: Proposal):
        super().__init__()
        self.inv_density = inv_density
        self.proposal = proposal

    def step(self, x: Tensor) -> Tensor:
        """
        Perform a single Metropolis step using the given propsal distribution.
        Args:
        - x: (bs, dim)
        Returns:
        - x_new: (bs, dim)
        """
        x_new = self.proposal(x)  # (bs, dim)
        log_ratio = (
            self.inv_density.log_density(x_new)
            - self.inv_density.log_density(x)
            + self.proposal.log_transition_factor(x, x_new)
        )  # (bs, 1)
        accept_mask = torch.logical_or(
            log_ratio >= 0,
            torch.rand_like(log_ratio).to(log_ratio) < torch.exp(log_ratio),
        )  # (bs, 1)
        accept_mask = accept_mask.expand_as(x)
        return torch.where(accept_mask, x_new, x)


def leapfrog_integrate_step(
    x: torch.Tensor,
    p: torch.Tensor,
    inv_density: Density,
    step_size: float,
    weight: float,
):
    p_mid = p - 0.5 * step_size * inv_density.energy_gradient(x)
    x_new = x + step_size * p_mid / weight
    p_new = p_mid - 0.5 * step_size * inv_density.energy_gradient(x_new)
    return x_new, p_new


class HMC(MCMC):
    def __init__(
        self,
        inv_density: Density,
        num_integration_steps: int,
        step_size: float = 0.1,
        weight: float = 1.0,
    ):
        super().__init__()
        self.inv_density = inv_density
        self.num_integration_steps = num_integration_steps
        self.step_size = step_size
        self.weight = weight

    @torch.no_grad()
    def step(self, x: Tensor) -> Tensor:
        """
        Args:
        - x: (bs, dim)
        Returns:
        - x_new: (bs, dim)
        """
        # Randomly sample momentum for batch
        p = self.weight * torch.randn_like(x)  # (bs, dim)
        log_h = torch.log(self.get_hamiltonian(x, p))  # (bs, 1)
        # Integrate
        x_prop, p_prop = x, p
        for _ in range(self.num_integration_steps):
            x_prop, p_prop = leapfrog_integrate_step(
                x_prop, p_prop, self.inv_density, self.step_size, self.weight
            )

        log_h_prop = torch.log(self.get_hamiltonian(x_prop, -p_prop))
        log_ratio = log_h - log_h_prop  # (bs, 1)

        accept_mask = torch.logical_or(
            log_ratio >= 0,
            torch.rand_like(log_ratio).to(log_ratio) < torch.exp(log_ratio),
        )  # (bs, 1)
        accept_mask = accept_mask.expand_as(x)
        return torch.where(accept_mask, x_prop, x)

    def get_hamiltonian(self, x: Tensor, p: Tensor) -> Tensor:
        """
        Args:
        - x: (bs, dim)
        Returns:
        - hamiltonian: (bs, dim)
        """
        potential_energy = self.inv_density.energy(x)  # (bs, 1)
        kinetic_energy = (
            0.5 * (p.pow(2)).sum(dim=-1, keepdims=True) / self.weight
        )  # (bs, 1)
        return potential_energy + kinetic_energy


class ParallelTempering(MCMC):
    """MCMC on joint density over temperature ladder"""

    pass
