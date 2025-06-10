from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from src.simulation.mcmc import MCMC
from src.systems.base import Sampleable
from src.systems.density_path import DensityPath


class MCMCSampleable(Sampleable):
    def __init__(self, mcmc: MCMC, source: Sampleable):
        super().__init__()
        self.register_module("mcmc", mcmc)
        self.register_module("source", source)

    @property
    def dim(self) -> int:
        return self.source.dim

    def sample_trajectory(self, num_steps) -> Tensor:
        x = self.source.sample(1)
        trajectory = [x]
        for _ in range(num_steps):
            x = self.mcmc.step(x)
            trajectory.append(x)
        return torch.stack(trajectory).transpose(0, 1)

    def sample(self, num_samples: int, num_steps: int) -> Tensor:
        x = self.source.sample(num_samples)
        for _ in tqdm(range(num_steps)):
            x = self.mcmc.step(x)
        return x


class AnnealedImportanceSampler(nn.Module):
    """
    Implementation of annealed importance sampling.
    TODO: Document implementation derivation. Basic idea is that when we simulate multiple MCMC steps at a fixed intermediate density during the annealing, the importance weights cancel out so that we need only update when t is updated.
    """

    def __init__(
        self,
        # This should ALSO be a density. Python type system makes this clunky.
        source: Sampleable,
        density_path: DensityPath,
        mcmc_class: Type[MCMC],
        **mcmc_kwargs,
    ):
        super().__init__()
        self.source = source
        self.density_path = density_path
        self.mcmc_class = mcmc_class
        self.mcmc_kwargs = mcmc_kwargs

    @property
    def start_sampleable(self) -> Sampleable:
        return self.density_path.start_sampleable

    @torch.no_grad()
    def sample_with_importance_weights(
        self,
        num_samples: int,
        num_timesteps: int = 100,
        num_steps_per_t: int = 1,
        record_trajectories: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[List[Tensor]]]:
        """
        Returns:
        - x: (bs, dim)
        - log_weights: (bs, 1)
        - trajectories: List of (bs, nts, dim) if record_trajectories else None
        """

        x = self.start_sampleable.sample(num_samples)  # (bs, dim)
        trajectories = [x.detach().clone()] if record_trajectories else None
        log_weights = torch.zeros_like(x[:, :1])  # (bs, 1)
        prev_density = None  # e.g., p_{t-\Delta t}
        curr_density = self.density_path.get_density_at(0)  # e.g., p_t
        for global_idx, t in tqdm(enumerate(torch.linspace(0, 1, num_timesteps)[1:])):
            # Update densities
            prev_density = curr_density
            curr_density = self.density_path.get_density_at(t)
            # Update log importance weights: this is the current x's contribution to log density ratio of forward and backward processes
            log_weights += curr_density.log_density(x) - prev_density.log_density(x)
            # Initialize MCMC at current density
            mcmc = self.mcmc_class(inv_density=curr_density, **self.mcmc_kwargs)
            # Simulate MCMC for num_steps_per_t steps
            for _ in range(num_steps_per_t):
                x = mcmc.step(x)
            if trajectories:
                trajectories.append(x.detach().clone())
        return x, log_weights, trajectories
