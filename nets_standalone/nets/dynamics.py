from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from .base import Sampleable
from .misc import record_every_idxs
from .vector_fields import VectorField


class AuxiliaryProcess(nn.Module, ABC):
    @abstractmethod
    def initial_value(self, num_samples: int) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def integrate_step(self, at: Tensor, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        raise NotImplementedError


class DivegenceAuxiliaryProcess(AuxiliaryProcess):
    def __init__(self, control: VectorField, divergence_mode: str):
        super().__init__()
        self.control = control
        self.divergence_mode = divergence_mode

    def initial_value(self, num_samples: int) -> Tensor:
        return torch.zeros(num_samples, 1)

    def integrate_step(self, at: Tensor, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        return at - self.control.divergence(xt, t, mode=self.divergence_mode) * dt


class ForwardProcess(nn.Module, ABC):
    def __init__(
        self,
        source: Optional[Sampleable] = None,
        auxiliary_processes: Optional[Dict[str, AuxiliaryProcess]] = None,
    ):
        super().__init__()
        self.source = source
        self.auxiliary_processes = nn.ModuleDict(auxiliary_processes or {})

    @abstractmethod
    def state_integrate_step(self, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        raise NotImplementedError

    def integrate_with_trajectory(self, x0: Tensor, aux0: Dict[str, Tensor], ts: Tensor, use_tqdm: bool) -> tuple[Tensor, Dict[str, Tensor], Tensor]:
        xt = x0
        auxt_dict = aux0
        x_trajectory = [xt.clone()]
        aux_trajectory = {k: [v.clone()] for k, v in auxt_dict.items()}

        step_iterable = tqdm(range(ts.shape[1] - 1)) if use_tqdm else range(ts.shape[1] - 1)
        for step in step_iterable:
            t = ts[:, step]
            nt = ts[:, step + 1]
            new_aux_dict = {}
            for key, aux_process in self.auxiliary_processes.items():
                updated = aux_process.integrate_step(auxt_dict[key], xt, t, nt - t)
                new_aux_dict[key] = updated
                aux_trajectory[key].append(updated.clone())
            auxt_dict = new_aux_dict
            xt = self.state_integrate_step(xt, t, nt - t)
            x_trajectory.append(xt.clone())

        return (
            torch.stack(x_trajectory, dim=1),
            {k: torch.stack(v, dim=1) for k, v in aux_trajectory.items()},
            ts,
        )

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        ts: Tensor,
        num_samples: Optional[int] = None,
        x0: Optional[Tensor] = None,
        use_tqdm: bool = False,
        record_every: int = 1,
    ) -> tuple[Tensor, Dict[str, Tensor], Tensor]:
        if x0 is None:
            if self.source is None or num_samples is None:
                raise ValueError("A source or explicit x0 is required.")
            x0 = self.source.sample(num_samples).to(ts)
        num_samples = x0.shape[0]
        aux0 = {k: aux.initial_value(num_samples).to(x0) for k, aux in self.auxiliary_processes.items()}
        x_trajectory, aux_trajectory, ts = self.integrate_with_trajectory(x0, aux0, ts, use_tqdm)
        if record_every > 1:
            record_idxs = record_every_idxs(ts.shape[1], record_every)
            x_trajectory = x_trajectory[:, record_idxs]
            aux_trajectory = {k: v[:, record_idxs] for k, v in aux_trajectory.items()}
            ts = ts[:, record_idxs]
        return x_trajectory, aux_trajectory, ts

    @torch.no_grad()
    def sample(
        self,
        ts: Tensor,
        num_samples: Optional[int] = None,
        x0: Optional[Tensor] = None,
        use_tqdm: bool = False,
    ) -> tuple[Tensor, Dict[str, Tensor]]:
        trajectory, aux_trajectory, _ = self.sample_with_trajectory(
            ts=ts,
            num_samples=num_samples,
            x0=x0,
            use_tqdm=use_tqdm,
            record_every=1,
        )
        return trajectory[:, -1], {k: v[:, -1] for k, v in aux_trajectory.items()}


class ForwardDiffusionProcess(ForwardProcess):
    @abstractmethod
    def drift(self, xt: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def noise(self, xt: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    def state_integrate_step(self, xt: Tensor, t: Tensor, dt: Tensor, clamp_val: float = 1000) -> Tensor:
        drift = torch.clamp(self.drift(xt, t), -clamp_val, clamp_val)
        noise = torch.randn_like(xt)
        return xt + drift * dt + self.noise(xt, t) * noise * torch.sqrt(torch.abs(dt))


class ODEProcess(ForwardDiffusionProcess):
    def __init__(self, source: Sampleable, vector_field: VectorField, auxiliary_processes: Optional[Dict[str, AuxiliaryProcess]] = None):
        super().__init__(source, auxiliary_processes)
        self.vector_field = vector_field

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.vector_field(x, t)

    def noise(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.zeros_like(x)
