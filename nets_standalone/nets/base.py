from __future__ import annotations

from abc import ABC
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, vmap

from .misc import get_module_device


class Sampleable(nn.Module, ABC):
    def sample(self, num_samples: int) -> Tensor:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def device(self) -> Optional[torch.device]:
        return get_module_device(self)


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


class WrapperDensity(Density):
    def __init__(self, log_density_fn: Callable[[Tensor], Tensor], dim: int):
        super().__init__()
        self._log_density_fn = log_density_fn
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def log_density(self, x: Tensor) -> Tensor:
        return self._log_density_fn(x)
