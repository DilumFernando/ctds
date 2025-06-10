import math
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, vmap

from src.utils.misc import get_device, get_module_device


class Sampleable(nn.Module):
    """
    Abstract base class for all sampleable distributions.
    """

    def sample(self, num_samples: int) -> Tensor:
        raise NotImplementedError()

    @property
    def dim(self) -> int:
        raise NotImplementedError()

    @property
    def device(self) -> Optional[torch.device]:
        return get_module_device(self)

    def sample_at_temp(self, beta: Tensor) -> Tensor:
        """
        Batched sampling at arbitrary temperatures.
        Args:
            beta: (bs, 1)
        Returns:
            x: (bs, dim)
        """
        raise NotImplementedError()
    
class WrapperSampleable(Sampleable):
    def __init__(self, dim: int, sample_fn: Callable):
        super().__init__()
        self._dim = dim
        self.sample_fn = sample_fn

    @property
    def dim(self) -> int:
        return self._dim
    
    def sample(self, num_samples: int) -> Tensor:
        return self.sample_fn(num_samples)
    
class ConstantSampleable(Sampleable):
    def __init__(self, x: Tensor):
        """
        Args:
        - x: (1, dim)
        """
        super().__init__()
        self.register_buffer("x", x)

    @property
    def dim(self) -> int:
        return self.x.shape[1]
    
    def sample(self, num_samples: int) -> Tensor:
        return self.x.expand(num_samples, -1).clone()    

class ProductSampleable(Sampleable):
    def __init__(self, sampleables: List[Sampleable]):
        super().__init__()
        self.sampleables = nn.ModuleList(sampleables)

    @property
    def dim(self) -> int:
        return sum([sampleable.dim for sampleable in self.sampleables])

    def sample(self, num_samples: int) -> Tensor:
        return torch.cat(
            [sampleable.sample(num_samples) for sampleable in self.sampleables], dim=1
        )

    def sample_at_temp(self, beta: Tensor) -> Tensor:
        """
        Batched sampling at arbitrary temperatures.
        Args:
            beta: (bs, 1)
        Returns:
            x: (bs, dim)
        """
        return torch.cat(
            [sampleable.sample_at_temp(beta) for sampleable in self.sampleables],
            dim=1,
        )


class Density(nn.Module):
    """
    Abstract base class for all densities.
    """

    def dim(self) -> int:
        raise NotImplementedError()

    @property
    def device(self) -> Optional[torch.device]:
        return get_module_device(self)

    def log_density(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
        Returns:
            log_density: (batch_size, 1)
        """
        raise NotImplementedError()

    def energy(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
        Returns:
            energy: (batch_size, 1)
        """
        raise NotImplementedError()

    def dx_log_density(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
        Returns:
            dx_log_density: (batch_size, dim)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        dx_log_density = vmap(jacrev(self.log_density))(x)  # (batch_size, 1, 1, 1, dim)
        return dx_log_density.squeeze((1, 2, 3))  # (batch_size, dim)

    def energy_gradient(self, x: Tensor) -> Tensor:
        return -self.dx_log_density(x)

    def raise_to_temperature(self, beta: float) -> "Density":
        raise NotImplementedError()


class WrapperDensity(Density):
    """
    Lightweight density wrapper which does not require duplicating weights.
    Intended for e.g., efficient implementation of annealed density path without.
    """

    def __init__(self, log_density_fn: Callable[[Tensor], Tensor], dim: int):
        super().__init__()
        self._log_density_fn = log_density_fn
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def log_density(self, x: Tensor) -> Tensor:
        return self._log_density_fn(x)

    def energy(self, x: Tensor) -> Tensor:
        return -self._log_density_fn(x)

    def raise_to_temperature(self, beta: float) -> "WrapperDensity":
        return WrapperDensity(lambda x: beta * self._log_density_fn(x), self._dim)


class BumpInterval(Sampleable, Density):
    """
    Bump function for the interval [a,b]
    """

    def __init__(self, low: float, high: float, sharpness: float):
        super().__init__()
        self.low = low
        self.high = high
        self.sharpness = sharpness
        self.register_buffer("device_dummy", torch.tensor(0.0))

    @property
    def dim(self) -> int:
        return 1

    def sample(self, num_samples: int) -> Tensor:
        """
        Rejection sample from the bump function.
        For numerical stability, we sample from [max(0.25, a - eps), b + eps] where eps = sqrt(log(100) / sharpness).
        """
        samples = torch.zeros(0, 1).to(self.device_dummy)
        sample_eps = math.sqrt(math.log(100) / self.sharpness)
        sample_lower = self.low - sample_eps
        sample_upper = self.high + sample_eps
        while samples.shape[0] < num_samples:
            x = (
                torch.rand(num_samples, 1).to(samples) * (sample_upper - sample_lower)
                + sample_lower
            )
            y = torch.rand_like(x)
            accept_idxs = torch.where(y < torch.exp(self.log_density(x)))[0]
            samples = torch.cat([samples, x[accept_idxs]])
        return samples[:num_samples]

    def log_density(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        return torch.where(
            x < self.low,
            -self.sharpness * ((x - self.low) ** 2),
            torch.where(
                x <= self.high,
                torch.tensor(0.0).to(x),
                -self.sharpness * ((x - self.high) ** 2),
            ),
        )
            

class CircularUniformFunction(Sampleable, Density):
    """
    Uniform function on 2D disk centered at the origin, for debugging purposes.
    """

    def __init__(self, r: float):
        super().__init__()
        self.r = r
        self.register_buffer("device_dummy", torch.tensor(0.0))

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> Tensor:
        """
        Rejection sample from the bump function.
        For numerical stability, we sample from [r - eps, r + eps] where eps = sqrt(log(100) / sharpness).
        """
        samples = torch.zeros(0, 2).to(self.device_dummy)
        sample_lower = -self.r
        sample_upper = self.r
        while samples.shape[0] < num_samples:
            x = (
                torch.rand(num_samples, 2).to(samples) * (sample_upper - sample_lower)
                + sample_lower
            )
            y = torch.rand_like(x)
            accept_idxs = torch.where(y < torch.exp(self.log_density(x)))[0]
            samples = torch.cat([samples, x[accept_idxs]])
        return samples[:num_samples]

    def log_density(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, 2)
        Returns:
            log_density: (batch_size, 1)
        """
        r = torch.norm(x, dim=1, keepdim=True)
        return torch.where(r < self.r, torch.tensor(0.0).to(x), -1000)

    def sample_at_temp(self, beta: Tensor) -> Tensor:
        """
        Batched sampling at arbitrary temperatures.
        Args:
            beta: (bs, 1)
        Returns:
            x: (bs, dim)
        """
        return self.sample(beta.shape[0])


class CircularBumpFunction(Sampleable, Density):
    """
    Bump function on 2D disk centered at the origin.
    """

    def __init__(self, r: float, sharpness: float):
        super().__init__()
        self.r = r
        self.sharpness = sharpness
        self.register_buffer("device_dummy", torch.tensor(0.0))

    @property
    def dim(self) -> int:
        return 2

    def sample(self, num_samples: int) -> Tensor:
        """
        Rejection sample from the bump function.
        For numerical stability, we sample from [r - eps, r + eps] where eps = sqrt(log(100) / sharpness).
        """
        samples = torch.zeros(0, 2).to(self.device_dummy)
        sample_eps = math.sqrt(math.log(100) / self.sharpness)
        sample_lower = -self.r - sample_eps
        sample_upper = self.r + sample_eps
        while samples.shape[0] < num_samples:
            x = (
                torch.rand(num_samples, 2).to(samples) * (sample_upper - sample_lower)
                + sample_lower
            )
            y = torch.rand_like(x)
            accept_idxs = torch.where(y < torch.exp(self.log_density(x)))[0]
            samples = torch.cat([samples, x[accept_idxs]])
        return samples[:num_samples]

    def log_density(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, 2)
        Returns:
            log_density: (batch_size, 1)
        """
        r = torch.norm(x, dim=1, keepdim=True)
        return torch.where(
            r < self.r,
            torch.tensor(0.0).to(x),
            -self.sharpness * ((r - self.r) ** 2),
        )


class Uniform(Sampleable, Density):
    """
    Uniform distribution on [low, high]^dim.
    """

    def __init__(self, low: float, high: float, dim: int):
        super().__init__()
        self._low = low
        self._high = high
        self._dim = dim
        self.register_buffer("device_dummy", torch.tensor(0.0))

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def get_device(self) -> Optional[torch.device]:
        return None

    def sample(self, num_samples: int) -> Tensor:
        return (
            torch.rand(num_samples, self.dim).to(self.device_dummy)
            * (self._high - self._low)
            + self._low
        )

    def log_density(self, x: Tensor) -> Tensor:
        in_bounds_mask = ((x >= self._low) & (x <= self._high)).all(
            dim=1
        )  # (batch_size,)
        log_density = (
            torch.where(
                in_bounds_mask,
                torch.tensor(0.0),
                torch.tensor(-float("inf")),
            )
            .to(x)
            .unsqueeze(-1)
        )  # (batch_size, 1)
        return log_density

    def energy(self, x: Tensor) -> Tensor:
        return -self.log_density(x)

    def raise_to_temperature(self, beta: float) -> "UniformDensity":
        raise NotImplementedError()
