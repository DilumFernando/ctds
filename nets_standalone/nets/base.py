from __future__ import annotations

from abc import ABC
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, vmap


def module_device(module: nn.Module) -> Optional[torch.device]:
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        pass
    return None


class Sampleable(nn.Module, ABC):
    def sample(self, num_samples: int) -> Tensor:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def device(self) -> Optional[torch.device]:
        return module_device(self)


class Density(nn.Module, ABC):
    @property
    def dim(self) -> int:
        raise NotImplementedError

    def log_density(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def dx_log_density(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        dx_log_density = vmap(jacrev(self.log_density))(x)
        return dx_log_density.squeeze((1, 2, 3))
