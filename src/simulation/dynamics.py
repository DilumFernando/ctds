import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, vmap
from tqdm import tqdm

from src.simulation.vector_field import VectorField
from src.systems.base import Sampleable
from src.utils.misc import get_module_device, record_every_idxs


class AuxiliaryProcess(nn.Module, ABC):
    """
    General-purpose class for Markovian auxliliary processes such as Jarzynski importance weights, continuous change of variable, etc.
    Here the auxiliary process is denoted by at
    """

    @abstractmethod
    def initial_value(self, num_samples: int) -> torch.Tensor:
        pass

    @abstractmethod
    def integrate_step(
        self, at: torch.Tensor, xt: torch.Tensor, t: Tensor, dt: Tensor
    ) -> torch.Tensor:
        pass


class DivegenceAuxiliaryProcess(AuxiliaryProcess):
    """
    Implements the change of variables formula
    """

    def __init__(self, control: VectorField, divergence_mode: str):
        super().__init__()
        self.control = control
        self.divergence_mode = divergence_mode

    def initial_value(self, num_samples: int) -> torch.Tensor:
        """
        Initial value for the auxiliary process
        Args:
        - num_samples: int
        Returns:
        - a0: (num_samples, 1)
        """
        return torch.zeros(num_samples, 1)

    def integrate_step(
        self, at: torch.Tensor, xt: torch.Tensor, t: Tensor, dt: Tensor
    ) -> torch.Tensor:
        """
        Integration step for the Jarzynski importance weights
        Args:
        - at: (batch_size, 1)
        - xt: (batch_size, dim)
        - t: (batch_size, 1)
        - dt: (batch_size, 1)
        Returns:
        - updated_at: (batch_size, 1)
        """
        div = self.control.divergence(xt, t, mode=self.divergence_mode)
        return at - div * dt


class ForwardProcess(nn.Module, ABC):
    """
    A Markovian forward process xt (e.g., a flow, diffusion, or jump process) along with some number of auxiliary processes
    """

    def __init__(
        self,
        source: Optional[Sampleable] = None,
        auxiliary_processes: Dict[str, AuxiliaryProcess] = {},
    ):
        super().__init__()
        if source is not None:
            self.register_module("source", source)
        else:
            self.source = None
        self.auxiliary_processes = nn.ModuleDict(auxiliary_processes)

    @property
    def dim(self) -> int:
        if self.source is not None:
            return self.source.dim
        else:
            raise NotImplementedError

    @abstractmethod
    def state_integrate_step(self, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        """
        Integration step for state xt
        Args:
        - xt: (batch_size, dim)
        - t: (batch_size, 1)
        Returns:
        - updated_xt: (batch_size, dim)
        """
        pass

    def integrate(
        self, x0: Tensor, aux0: Dict[str, Tensor], ts: Tensor, use_tqdm: bool = False
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Integrate the forward diffusion process from x_0 to x_t. Less memory-intensive than integrate_with_trajectory.
        Args:
        - x0: (batch_size, dim)
        - aux0: str -> (batch_size, aux_dim)
        - ts: (batch_size, num_timesteps, 1)
        Returns:
        - xt: (batch_size, dim)
        - auxt_dict: str -> (batch_size, aux_dim)
        """
        # Initialize the states
        xt = x0
        auxt_dict = aux0

        _, num_timesteps, _ = ts.shape
        step_iterable = (
            tqdm(range(num_timesteps - 1)) if use_tqdm else range(num_timesteps - 1)
        )
        for step in step_iterable:
            t = ts[:, step]
            nt = ts[:, step + 1]
            # Update auxiliary processes
            new_aux_dict = {}
            for k, aux_process in self.auxiliary_processes.items():
                auxt = auxt_dict[k]
                new_aux_dict[k] = aux_process.integrate_step(auxt, xt, t, nt - t)
            auxt_dict = new_aux_dict
            # Update state process
            xt = self.state_integrate_step(xt, t, nt - t)

        return (xt, auxt_dict)

    def integrate_with_trajectory(
        self, x0: Tensor, aux0: Dict[str, Tensor], ts: Tensor, use_tqdm: bool = False
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """
        Integrate the forward diffusion process from x_0 to x_t and return the trajectory.
        Args:
        - x0: (batch_size, dim)
        - aux0: str -> (batch_size, aux_dim)
        - ts: (batch_size, num_timesteps, 1)
        Returns:
        - x_trajectory: (batch_size, num_timesteps, dim)
        - aux_trajectory: str -> (batch_size, num_timesteps, aux_dim)
        - ts: (batch_size, num_timesteps, 1)
        """
        # Initialize the states
        xt = x0
        auxt_dict = aux0

        # Initialize the trajectories
        x_trajectory = [xt.clone()]
        aux_trajectory = {k: [v.clone()] for k, v in auxt_dict.items()}

        _, num_timesteps, _ = ts.shape
        step_iterable = (
            tqdm(range(num_timesteps - 1)) if use_tqdm else range(num_timesteps - 1)
        )
        for step in step_iterable:
            t = ts[:, step]
            nt = ts[:, step + 1]
            # Update auxiliary processes
            new_aux_dict = {}
            for k, aux_process in self.auxiliary_processes.items():
                auxt = auxt_dict[k]
                new_aux_dict[k] = aux_process.integrate_step(auxt, xt, t, nt - t)
                aux_trajectory[k].append(new_aux_dict[k].clone())
            auxt_dict = new_aux_dict
            # Update state process
            xt = self.state_integrate_step(xt, t, nt - t)
            x_trajectory.append(xt.clone())

        x_trajectory = torch.stack(x_trajectory, dim=1)
        aux_trajectory = {k: torch.stack(v, dim=1) for k, v in aux_trajectory.items()}
        return x_trajectory, aux_trajectory, ts

    @torch.no_grad()
    def sample(
        self,
        ts: Tensor,
        num_samples: Optional[int] = None,
        x0: Optional[Tensor] = None,
        use_tqdm: bool = False
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Sample from the forward diffusion process and return the terminal samples. Less memory-intensive than sample_with_trajectory.
        Args:
        - ts: (bs, num_timesteps, 1)
        - x0: (bs, dim)
        Returns:
        - x_final: (bs, dim)
        - auxt_dict_final: str -> (bs, aux_dim)
        """
        # Sample x0 if not provided
        if x0 is None:
            assert num_samples is not None
            assert self.source is not None
            x0 = self.source.sample(num_samples)
        num_samples = x0.shape[0]

        # Initialize auxiliary processes
        aux0 = {
            k: aux.initial_value(num_samples).to(x0)
            for k, aux in self.auxiliary_processes.items()
        }

        # Integrate
        return self.integrate(x0, aux0, ts, use_tqdm=use_tqdm)

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        ts: Tensor,
        num_samples: Optional[int] = None,
        x0: Optional[Tensor] = None,
        use_tqdm: bool = False,
        record_every: int = 1,
    ) -> Tuple[Tensor, Dict[str, Tensor], Tensor]:
        """
        Sample from the forward diffusion process and return the trajectory.
        Args:
        - ts: (bs, num_timesteps, 1)
        - x0: (bs, dim)
        Returns:
        - x_a_trajectory: (batch_size, num_timesteps, dim + 1)
        - aux_trajectory: str -> (batch_size, num_timesteps, aux_dim)
        - ts: (batch_size, num_timesteps, 1)
        """
        # Sample x0 if not provided
        if x0 is None:
            assert num_samples is not None
            assert self.source is not None
            x0 = self.source.sample(num_samples)
        num_samples = x0.shape[0]

        # Initialize auxiliary processes
        aux0 = {
            k: aux.initial_value(num_samples).to(x0)
            for k, aux in self.auxiliary_processes.items()
        }

        # Integrate
        x_trajectory, aux_trajectory, ts = self.integrate_with_trajectory(
            x0, aux0, ts, use_tqdm=use_tqdm
        )

        # Record every if necessary
        if record_every > 1:
            record_idxs = record_every_idxs(ts.shape[1], record_every)
            x_trajectory = x_trajectory[:, record_idxs]
            aux_trajectory = {k: v[:, record_idxs] for k, v in aux_trajectory.items()}
            ts = ts[:, record_idxs]
        return x_trajectory, aux_trajectory, ts


class ForwardJumpProcess(ForwardProcess):
    pass


class ForwardDiffusionProcess(ForwardProcess):
    """
    Ito diffusion dx_t = drift(xt,t) dt + noise(xt,t) dW_t
    """

    def __init__(
        self,
        source: Optional[Sampleable] = None,
        auxiliary_processes: Dict[str, AuxiliaryProcess] = {},
    ):
        super().__init__(source, auxiliary_processes)

    def drift(self, xt: Tensor, t: Tensor) -> Tensor:
        """
        Args:
        - xt: (batch_size, dim)
        - t: (batch_size, 1)
        Returns:
        - drift: (batch_size, dim)
        """
        raise NotImplementedError

    def noise(self, xt: Tensor, t: Tensor) -> Tensor:
        """
        Args:
        - xt: (batch_size, dim)
        - t: (batch_size, 1)
        Returns:
        - noise: (batch_size, dim)
        """
        raise NotImplementedError

    def state_integrate_step(
        self, xt: Tensor, t: Tensor, dt: Tensor, clamp_val: float = 1000
    ) -> Tensor:
        """
        Euler-Maruyama integration step for state xt
        Args:
        - xt: (batch_size, dim)
        - t: (batch_size, 1)
        - dt: (batch_size, 1)
        Returns:
        - updated_xt: (batch_size, dim)
        """
        drift = self.drift(xt, t)  # (batch_size, dim)
        drift = torch.clamp(drift, -clamp_val, clamp_val)
        noise_coeff = self.noise(xt, t)  # (batch_size, dim)
        noise = torch.randn_like(xt)  # (batch_size, dim)
        return xt + drift * dt + noise_coeff * noise * torch.sqrt(torch.abs(dt))


class BrownianMotion(ForwardDiffusionProcess):
    """
    Implements the Brownian motion dx_t = noise(t) dW_t
    """

    def __init__(self, source: Sampleable, auxiliary_processes: Dict[str, AuxiliaryProcess] = {}):
        super().__init__(source, auxiliary_processes)

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.zeros_like(x)
    
    def noise(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.ones_like(x)


class ODEProcess(ForwardDiffusionProcess):
    """
    Implements an ODE dx_t = drift(xt,t) dt
    """

    def __init__(self, source: Sampleable, vector_field: VectorField, auxiliary_processes: Dict[str, AuxiliaryProcess] = {}):
        super().__init__(source, auxiliary_processes)
        self.vector_field = vector_field  

    def drift(self, x: Tensor, t: Tensor):
        return self.vector_field(x, t)
    
    def noise(self, x: Tensor, t: Tensor):
        return torch.zeros_like(x)


class ForwardProcessSampleable(Sampleable):
    def __init__(self, forward_process: ForwardProcess, ts: Tensor):
        """
        Args:
        - forward_process: ForwardProcess
        - ts: (num_timesteps,)
        """
        super().__init__()
        self.forward_process = forward_process
        self.register_buffer("ts", ts)

    @property
    def dim(self) -> int:
        return self.forward_process.source.dim

    def sample(self, num_samples: int, use_tqdm: bool = True, **kwargs) -> Tensor:
        x1, _ = self.forward_process.sample(
            self.ts.view(1, -1, 1).expand(num_samples, -1, 1),
            num_samples=num_samples,
            use_tqdm=use_tqdm,
            **kwargs,
        )
        return x1