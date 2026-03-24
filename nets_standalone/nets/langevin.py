from __future__ import annotations

import math
from typing import Dict

import torch
from torch import Tensor

from .density_paths import DensityPath
from .dynamics import AuxiliaryProcess, ForwardDiffusionProcess
from .vector_fields import VectorField


class AnnealedOverdampedLangevin(ForwardDiffusionProcess):
    def __init__(
        self,
        control: VectorField,
        damping: float,
        density_path: DensityPath,
        auxiliary_processes: Dict[str, AuxiliaryProcess],
    ):
        super().__init__(density_path.start_sampleable, auxiliary_processes)
        self.register_module("density_path", density_path)
        self.register_module("control", control)
        self.damping = damping

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        return self.control(x, t) + self.density_path.dx_log_density(x, t) * self.damping

    def noise(self, x: Tensor, t: Tensor) -> Tensor:
        return math.sqrt(2 * self.damping) * torch.ones_like(x)
