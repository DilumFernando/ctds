from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.distributions as D
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.func import jacrev, vmap

from src.utils.nn import FeedForward, GaussianFourierEncoder
from src.systems.base import Density, Sampleable, WrapperDensity
from src.systems.distributions import GMM, Gaussian, GaussianGMM, GMMDensity
from src.utils.misc import cuda_profile

#################
# Density Paths #
#################


class DensityPath(nn.Module, ABC):
    """
    Parameterizes a density path p: [0,1] -> Density.
    Note that we distinguish between the starting sampleable and the starting density in order to decouple these two - they are the same up to normalizing constant but in many situations it is useful to (either implicitly or explicitly) use an alternate normalizing constant.
    """

    @property
    def dim(self) -> int:
        return self.start_sampleable.dim

    @property
    @abstractmethod
    def start_sampleable(self) -> Sampleable:
        """
        Returns a sampleable for the source density. In some cases, this is not possible, in which case this method will just raise an error
        """
        raise NotImplementedError

    @property
    def end_sampleable(self) -> Sampleable:
        """
        Returns a sampleable for the target density. In some cases, this is not possible, in which case this method will just raise an error
        """
        raise NotImplementedError

    def get_samples_at(self, t: Tensor) -> Tensor:
        """
        To be implemented in the special case that intermediate samples can be generated
        Args:
            t: (batch_size, 1)
        """
        raise NotImplementedError

    @abstractmethod
    def get_density_at(self, t: float) -> Density:
        pass

    @abstractmethod
    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        pass

    def density_at(self, x: Tensor, t: Tensor) -> Density:
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            density: Density
        """
        return torch.exp(self.log_density(x, t))

    @cuda_profile
    def dt_log_density(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            dt_log_density: (batch_size, 1)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, 1)
        t = t.unsqueeze(1)  # (batch_size, 1, 1)
        dt_log_density = vmap(jacrev(self.log_density, argnums=1))(
            x, t
        )  # (batch_size, 1, 1, 1, 1)
        return dt_log_density.squeeze((2, 3, 4))  # (batch_size, 1)

    def energy_at(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            energy_at: (batch_size, 1)
        """
        return -self.log_density(x, t)

    @cuda_profile
    def dx_log_density(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            dx_log_density: (batch_size, 1)
        """
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        t = t.unsqueeze(1)  # (batch_size, 1, 1)
        dx_log_density = vmap(jacrev(self.log_density, argnums=0))(
            x, t
        )  # (batch_size, 1, 1, 1, dim)
        return dx_log_density.squeeze((1, 2, 3))  # (batch_size, dim)

    def dx_energy_at(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            dx_energy: (batch_size, 1)
        """
        return -self.dx_log_density(x, t)


class WrapperDensityPath(DensityPath):
    def __init__(self, sampleable: Sampleable, log_density_fn: Callable):
        super().__init__()
        self.sampleable = sampleable
        self.log_density_fn = log_density_fn

    @property
    def start_sampleable(self) -> Sampleable:
        return self.sampleable

    def get_density_at(self, t: float) -> Density:
        def log_density_fn(x: Tensor) -> Tensor:
            tt = t * torch.ones(x.shape[0], 1).to(x)
            return self.log_density(x, tt)

        return WrapperDensity(log_density_fn, self.sampleable.dim)

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (bs, dim)
            t: (bs, 1)
        Returns:
            log_density: (bs, 1)
        """
        return self.log_density_fn(x, t)

    @classmethod
    def raise_to_temperature(
        cls, density_path: DensityPath, beta: float
    ) -> "DensityPath":
        def log_density_fn(x, t):
            return density_path.log_density(x, t) * beta

        return cls(
            sampleable=density_path.start_sampleable.raise_to_temperature(beta),
            log_density_fn=log_density_fn,
        )


class ConstantDensityPath(DensityPath):
    def __init__(self, sampleable: Sampleable, density: Density):
        super().__init__()
        self.register_module("sampleable", sampleable)
        self.register_module("density", density)

    @property
    def start_sampleable(self) -> Sampleable:
        return self.sampleable

    def get_density_at(self, t: float) -> Density:
        return self.density

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        return self.density.log_density(x)


class LinearDensityPath(DensityPath):
    """
    Density path [0,1] -> Density parameterized by a geometric average of a source and target density.
    """

    def __init__(
        self,
        start_sampleable: Sampleable,
        start: Density,
        end: Density,
        end_sampleable: Optional[Sampleable] = None,
    ):
        super().__init__()
        self.register_module("_start_sampleable", start_sampleable)
        self.register_module("start", start)
        self.register_module("end", end)
        if end_sampleable is not None:
            self.register_module("_end_sampleable", end_sampleable)

    @property
    def start_sampleable(self) -> Sampleable:
        return self._start_sampleable

    @property
    def end_sampleable(self) -> Sampleable:
        if hasattr(self, "_end_sampleable"):
            return self._end_sampleable
        else:
            raise ValueError("End sampleable not defined.")

    def get_density_at(self, t: Union[Tensor, float]) -> Density:
        log_density_fn = partial(self.log_density, t=t)
        return WrapperDensity(log_density_fn, self.dim)

    def log_density(self, x: Tensor, t: Tensor):
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        start_log_density = self.start.log_density(x)  # (batch_size, 1)
        end_log_density = self.end.log_density(x)  # (batch_size, 1)
        # (batch_size, 1)
        return t * end_log_density + (1 - t) * start_log_density


class LearnableLinearDensityPath(LinearDensityPath):
    """
    Density path [0,1] -> Density parameterized by a geometric average of a source and target density plus learned compononent as in https://arxiv.org/abs/2301.07388
    """

    def __init__(
        self,
        start_sampleable: Sampleable,
        start: Density,
        end: Density,
        learnable_hiddens: List[int],
        use_fourier: bool,
        x_fourier_dim: Optional[int] = None,
        x_fourier_sigma: Optional[float] = None,
        t_fourier_dim: Optional[int] = None,
        t_fourier_sigma: Optional[float] = None,
        end_sampleable: Optional[Sampleable] = None,
    ):
        super().__init__(start_sampleable, start, end, end_sampleable)
        self.use_fourier = use_fourier
        if self.use_fourier:
            assert x_fourier_dim is not None and t_fourier_dim is not None
            assert x_fourier_sigma is not None and t_fourier_sigma is not None
            input_dim = x_fourier_dim + t_fourier_dim
            self.mlp = FeedForward([input_dim] + learnable_hiddens + [1])
            self.x_encoder = GaussianFourierEncoder(self.dim, x_fourier_dim, x_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)

            def fwd(x: Tensor, t: Tensor) -> Tensor:
                x_encoding = self.x_encoder(x)
                t_encoding = self.t_encoder(t)
                xt_encoding = torch.cat([x_encoding, t_encoding], dim=-1)
                return self.mlp(xt_encoding)

            self.learned_potential = fwd
        else:
            self.mlp = FeedForward([self.dim + 1] + learnable_hiddens + [1])

            def fwd(x: Tensor, t: Tensor) -> Tensor:
                xt = torch.cat([x, t], dim=-1)
                return self.mlp(xt)
            
            self.learned_potential = fwd

    @property
    def start_sampleable(self) -> Sampleable:
        return self._start_sampleable

    @property
    def end_sampleable(self) -> Sampleable:
        if hasattr(self, "_end_sampleable"):
            return self._end_sampleable
        else:
            raise ValueError("End sampleable not defined.")

    def get_density_at(self, t: Union[Tensor, float]) -> Density:
        log_density_fn = partial(self.log_density, t=t)
        return WrapperDensity(log_density_fn, self.dim)

    def log_density(self, x: Tensor, t: Tensor):
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        geometric_component = super().log_density(x, t)
        learned_component = t * (1 - t) * self.learned_potential(x, t)
        return geometric_component + learned_component


class LinearGMMInterpolation(DensityPath, torch.nn.Module):
    def __init__(
        self,
        start_sampleable: Sampleable,
        start: GMMDensity,
        end: GMMDensity,
        end_sampleable: Optional[Sampleable] = None,
        log_density_shim: Optional[Callable] = None,
    ):
        super().__init__()
        assert start.nmodes == end.nmodes
        self.register_module("_start_sampleable", start_sampleable)
        self.register_module("start", start)
        self.register_module("end", end)
        if end_sampleable is not None:
            self.register_module("_end_sampleable", end_sampleable)
        self.log_density_shim = log_density_shim

    @property
    def start_sampleable(self) -> Sampleable:
        return self._start_sampleable

    @property
    def end_sampleable(self) -> Sampleable:
        if hasattr(self, "_end_sampleable"):
            return self._end_sampleable
        else:
            raise ValueError("End sampleable not defined.")

    def mu_t(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (1,)
        Returns:
            mu_t: (nmodes, dim,)
        """
        return (1 - t) * self.start.means + t * self.end.means

    def sigma_t(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (1,)
        Returns:
            sigma_t: (nmodes, dim, dim)
        """
        return (1 - t) * self.start.covs + t * self.end.covs

    def get_w_t(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (1,)
        Returns:
            w_t: (nmodes,1)
        """
        return (1 - t) * self.start.weights + t * self.end.weights

    def get_density_at(self, t: Union[Tensor, float]) -> Density:
        log_density_fn = partial(self.log_density, t=t)
        return WrapperDensity(log_density_fn, self.dim)

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        mu_t = torch.vmap(self.mu_t)(t)  # (batch_size, nmodes, dim)
        sigma_t = torch.vmap(self.sigma_t)(t)  # (batch_size, nmodes, dim, dim)
        w_t = torch.vmap(self.get_w_t)(t)  # (batch_size, nmodes, 1)
        p_t = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(probs=w_t, validate_args=False),
            component_distribution=D.MultivariateNormal(
                loc=mu_t,
                covariance_matrix=sigma_t,
                validate_args=False,
            ),
            validate_args=False,
        )
        result = p_t.log_prob(x).unsqueeze(-1)
        if self.log_density_shim is not None:
            result = result + self.log_density_shim(x, t)  # (batch_size, 1)
        return result


class LinearGaussianInterpolation(DensityPath, torch.nn.Module):
    def __init__(self, start: Gaussian, end: Gaussian):
        super().__init__()
        self.register_module("start", start)
        self.register_module("end", end)

    @property
    def start_sampleable(self) -> Sampleable:
        return self.start

    @property
    def end_sampleable(self) -> Sampleable:
        return self.end

    def get_samples_at(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (batch_size, 1)
        Returns:
            samples: (batch_size, dim)
        """
        mu_t = torch.vmap(self.get_mu_t)(t)
        sigma_t = torch.vmap(self.get_sigma_t)(t)
        p_t = MultivariateNormal(mu_t, sigma_t, validate_args=False)
        return p_t.sample()

    @property
    def mu_0(self) -> Tensor:
        return self.start.mean  # (dim,)

    @property
    def sigma_0(self) -> Tensor:
        return self.start.cov  # (dim, dim)

    @property
    def mu_1(self) -> Tensor:
        return self.end.mean  # (dim,)

    @property
    def sigma_1(self) -> Tensor:
        return self.end.cov  # (dim, dim)

    def get_mu_t(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (1,)
        Returns:
            mu_t: (dim,)
        """
        return (1 - t) * self.mu_0 + t * self.mu_1  # (..., dim)

    def get_dt_mu_t(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (1,)
        Returns:
            dt_mu_t: (dim,)
        """
        return self.mu_1 - self.mu_0

    def get_sigma_t(self, t: Tensor) -> Tensor:
        """
        t: (1,)
        """
        return (1 - t) * self.sigma_0 + t * self.sigma_1  # (..., dim, dim)

    def get_sigma_t_cholesky(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (1,)
        Returns:
            cholesky: (dim, dim)
        """
        return torch.linalg.cholesky(self.get_sigma_t(t))

    def get_density_at(self, t: Tensor) -> Density:
        """
        Args:
            t: (1,)
        """
        mu_t = self.get_mu_t(t)
        sigma_t = self.get_sigma_t(t)
        return Gaussian(mu_t, sigma_t)

    def get_marginal_at(self, t: Tensor) -> MultivariateNormal:
        """
        Args:
            t: (batch_size, t)
        """
        mu_t = torch.vmap(self.get_mu_t)(t)  # (batch_size, dim)
        sigma_t = torch.vmap(self.get_sigma_t)(t)  # (batch_size, dim)
        return MultivariateNormal(mu_t, sigma_t, validate_args=False)

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        mu_t = torch.vmap(self.get_mu_t)(t)  # (batch_size, dim)
        sigma_t = torch.vmap(self.get_sigma_t)(t)  # (batch_size, dim)
        p_t = MultivariateNormal(mu_t, sigma_t, validate_args=False)
        return p_t.log_prob(x).unsqueeze(-1)


def build_density_path(cfg: DictConfig) -> DensityPath:
    x_dim = cfg.x_dim
    # Construct target
    if cfg.target == "fab_gmm":
        target = GMM.FAB_GMM(cov_scale=cfg.cov_scale)
    elif cfg.target == "symmetric_gmm_2D":
        target = GMM.symmetric_2D(cfg.target_nmodes, cfg.target_scale, cfg.target_std)
    elif cfg.target == "random_gmm":
        target = GMM.random(
            cfg.target_nmodes,
            cfg.target_scale,
            cfg.x_dim,
            cfg.target_std,
            cfg.target_seed,
        )
    else:
        raise NotImplementedError(f"Target {cfg.target} not implemented.")

    # Construct density path and source
    if cfg.density_path == "interpolant":
        assert isinstance(target, GMM)
        source = GaussianGMM(x_dim, cfg.source_std, target.nmodes)
        density_path = LinearGMMInterpolation(
            start_sampleable=source, start=source, end=target, end_sampleable=target
        )
    elif cfg.density_path == "linear":
        source = Gaussian.isotropic(dim=cfg.x_dim, std=cfg.source_std)
        density_path = LinearDensityPath(
            start_sampleable=source, start=source, end=target, end_sampleable=target
        )
    elif cfg.density_path == "learnable_linear":
        source = Gaussian.isotropic(dim=cfg.x_dim, std=cfg.source_std)
        if cfg.use_fourier:
            density_path = LearnableLinearDensityPath(
                start_sampleable=source,
                start=source,
                end=target,
                learnable_hiddens=cfg.learnable_hiddens,
                use_fourier=True,
                x_fourier_dim=cfg.x_fourier_dim,
                x_fourier_sigma=cfg.x_fourier_sigma,
                t_fourier_dim=cfg.t_fourier_dim,
                t_fourier_sigma=cfg.t_fourier_sigma,
                end_sampleable=target,
            )
        else:
            density_path = LearnableLinearDensityPath(
                start_sampleable=source,
                start=source,
                end=target,
                learnable_hiddens=cfg.learnable_hiddens,
                use_fourier=False,
                end_sampleable=target,
            )
    else:
        raise NotImplementedError(f"Density path {cfg.density_path} not implemented.")
    return density_path
