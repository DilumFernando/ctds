from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, vmap

from .base import Density, Sampleable
from .config import Config
from .distributions import GMM, Gaussian, WarmStartGMMPrior
from .nn import FeedForward, GaussianFourierEncoder


def cuda_profile(fn):
    @wraps(fn)
    def wrapper(*args, profile: bool = False, **kwargs):
        if profile and torch.cuda.is_available():
            start_bytes = torch.cuda.memory_allocated()
            result = fn(*args, **kwargs)
            end_bytes = torch.cuda.memory_allocated()
            gib = (end_bytes - start_bytes) / (1024 * 1024 * 1024)
            print(f"Call to {fn.__name__} used {gib:.3f} GiB of memory")
            return result
        return fn(*args, **kwargs)

    return wrapper


class DensityPath(nn.Module, ABC):
    @property
    def dim(self) -> int:
        return self.start_sampleable.dim

    @property
    @abstractmethod
    def start_sampleable(self) -> Sampleable:
        raise NotImplementedError

    @property
    def end_sampleable(self) -> Sampleable:
        raise NotImplementedError

    @abstractmethod
    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError

    @cuda_profile
    def dt_log_density(self, x: Tensor, t: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        dt_log_density = vmap(jacrev(self.log_density, argnums=1))(x, t)
        return dt_log_density.squeeze((2, 3, 4))

    @cuda_profile
    def dx_log_density(self, x: Tensor, t: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        t = t.unsqueeze(1)
        dx_log_density = vmap(jacrev(self.log_density, argnums=0))(x, t)
        return dx_log_density.squeeze((1, 2, 3))


class LinearDensityPath(DensityPath):
    def __init__(
        self,
        start_sampleable: Sampleable,
        start: Density,
        end: Density,
        end_sampleable: Sampleable,
    ):
        super().__init__()
        self.register_module("_start_sampleable", start_sampleable)
        self.register_module("start", start)
        self.register_module("end", end)
        self.register_module("_end_sampleable", end_sampleable)

    @property
    def start_sampleable(self) -> Sampleable:
        return self._start_sampleable

    @property
    def end_sampleable(self) -> Sampleable:
        return self._end_sampleable

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        return t * self.end.log_density(x) + (1 - t) * self.start.log_density(x)


class LearnableLinearDensityPath(LinearDensityPath):
    def __init__(
        self,
        start_sampleable: Sampleable,
        start: Density,
        end: Density,
        learnable_hiddens: list[int],
        use_fourier: bool,
        x_fourier_dim: int,
        x_fourier_sigma: float,
        t_fourier_dim: int,
        t_fourier_sigma: float,
        end_sampleable: Sampleable,
    ):
        super().__init__(start_sampleable, start, end, end_sampleable)
        if use_fourier:
            input_dim = x_fourier_dim + t_fourier_dim
            self.mlp = FeedForward([input_dim] + learnable_hiddens + [1])
            self.x_encoder = GaussianFourierEncoder(self.dim, x_fourier_dim, x_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)

            def learned_potential(x: Tensor, t: Tensor) -> Tensor:
                return self.mlp(torch.cat([self.x_encoder(x), self.t_encoder(t)], dim=-1))

        else:
            self.mlp = FeedForward([self.dim + 1] + learnable_hiddens + [1])

            def learned_potential(x: Tensor, t: Tensor) -> Tensor:
                return self.mlp(torch.cat([x, t], dim=-1))

        self.learned_potential = learned_potential

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        geometric_component = super().log_density(x, t)
        learned_component = t * (1 - t) * self.learned_potential(x, t)
        return geometric_component + learned_component


class WarmStartPotentialDensityPath(DensityPath):
    def __init__(
        self,
        prior: WarmStartGMMPrior,
        target: Density,
        target_sampleable: Sampleable,
    ):
        super().__init__()
        self.register_module("prior", prior)
        self.register_module("target", target)
        self.register_module("_start_sampleable", prior.start_sampleable())
        self.register_module("_end_sampleable", target_sampleable)

    @property
    def start_sampleable(self) -> Sampleable:
        return self._start_sampleable

    @property
    def end_sampleable(self) -> Sampleable:
        return self._end_sampleable

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        prior_component = self.prior.log_density(x, t)
        if self.prior.fixed_beta:
            prior_component = (1 - t) * prior_component
        return prior_component + t * self.target.log_density(x)


class LearnableWarmStartPotentialDensityPath(WarmStartPotentialDensityPath):
    def __init__(
        self,
        prior: WarmStartGMMPrior,
        target: Density,
        target_sampleable: Sampleable,
        learnable_hiddens: list[int],
        use_fourier: bool,
        x_fourier_dim: int,
        x_fourier_sigma: float,
        t_fourier_dim: int,
        t_fourier_sigma: float,
    ):
        super().__init__(prior=prior, target=target, target_sampleable=target_sampleable)
        if use_fourier:
            input_dim = x_fourier_dim + t_fourier_dim
            self.mlp = FeedForward([input_dim] + learnable_hiddens + [1])
            self.x_encoder = GaussianFourierEncoder(self.dim, x_fourier_dim, x_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)

            def learned_potential(x: Tensor, t: Tensor) -> Tensor:
                return self.mlp(torch.cat([self.x_encoder(x), self.t_encoder(t)], dim=-1))

        else:
            self.mlp = FeedForward([self.dim + 1] + learnable_hiddens + [1])

            def learned_potential(x: Tensor, t: Tensor) -> Tensor:
                return self.mlp(torch.cat([x, t], dim=-1))

        self.learned_potential = learned_potential

    def log_density(self, x: Tensor, t: Tensor) -> Tensor:
        base_component = super().log_density(x, t)
        learned_component = t * (1 - t) * self.learned_potential(x, t)
        return base_component + learned_component


def build_density_path(cfg: Config) -> DensityPath:
    if cfg.target == "fab_gmm":
        target = GMM.FAB_GMM(cov_scale=cfg.cov_scale)
    elif cfg.target == "symmetric_gmm_2d":
        target = GMM.symmetric_2d(
            nmodes=int(cfg.target_nmodes),
            scale=float(cfg.target_scale),
            std=float(cfg.target_std),
        )
    elif cfg.target == "random_gmm":
        target = GMM.random_gmm(
            nmodes=int(cfg.target_nmodes),
            scale=float(cfg.target_scale),
            dim=int(cfg.x_dim),
            std=float(cfg.target_std),
            seed=int(cfg.target_seed),
        )
    elif cfg.target == "asymmetric_two_mode_gmm":
        target = GMM.asymmetric_two_mode(
            dim=int(cfg.x_dim),
            mode_distance=float(cfg.target_mode_distance),
            small_mode_weight=float(cfg.target_small_mode_weight),
            large_mode_weight=float(cfg.target_large_mode_weight),
            small_mode_std=float(cfg.target_small_mode_std),
            large_mode_std=float(cfg.target_large_mode_std),
            randomize_mode_locations=bool(cfg.get("target_randomize_mode_locations", False)),
            seed=int(cfg.get("target_seed", cfg.seed)),
        )
    elif cfg.target == "constrained_random_gmm":
        target = GMM.constrained_random_gmm(
            dim=int(cfg.x_dim),
            mode_weights=[float(x) for x in cfg.target_mode_weights],
            mode_stds=[float(x) for x in cfg.target_mode_stds],
            min_mode_distance=float(cfg.target_min_mode_distance),
            seed=int(cfg.get("target_seed", cfg.seed)),
        )
    else:
        raise NotImplementedError(
            "Standalone repo only includes fab_gmm, symmetric_gmm_2d, random_gmm, asymmetric_two_mode_gmm, and constrained_random_gmm targets."
        )

    source = Gaussian.isotropic(dim=cfg.x_dim, std=cfg.source_std)
    if cfg.density_path == "linear":
        return LinearDensityPath(
            start_sampleable=source,
            start=source,
            end=target,
            end_sampleable=target,
        )
    if cfg.density_path == "learnable_linear":
        return LearnableLinearDensityPath(
            start_sampleable=source,
            start=source,
            end=target,
            learnable_hiddens=list(cfg.learnable_hiddens),
            use_fourier=bool(cfg.use_fourier),
            x_fourier_dim=int(cfg.x_fourier_dim),
            x_fourier_sigma=float(cfg.x_fourier_sigma),
            t_fourier_dim=int(cfg.t_fourier_dim),
            t_fourier_sigma=float(cfg.t_fourier_sigma),
            end_sampleable=target,
        )
    if cfg.density_path in {
        "warmstart_potential_sum",
        "warmstart_potential_sum_fixed_beta",
        "learnable_warmstart_potential_sum",
        "learnable_warmstart_potential_sum_fixed_beta",
    }:
        prior_warmstart_means = cfg.get("prior_warmstart_means")
        if prior_warmstart_means is None:
            target_means = getattr(target, "means", None)
            if target_means is None:
                raise ValueError(
                    "prior_warmstart_means must be provided unless the target exposes Gaussian-mixture means."
                )
            warmstart_means = target_means.detach().clone().to(dtype=torch.float32)
        else:
            warmstart_means = torch.tensor(prior_warmstart_means, dtype=torch.float32)
        if warmstart_means.ndim != 2 or warmstart_means.shape[1] != int(cfg.x_dim):
            raise ValueError("prior_warmstart_means must have shape [nmodes, x_dim].")
        prior_weights = cfg.get("prior_mode_weights")
        prior = WarmStartGMMPrior(
            means=warmstart_means,
            beta_max=float(cfg.get("prior_beta_max", 10.0)),
            beta_decay_rate=float(cfg.get("prior_beta_decay_rate", 15.0)),
            weights=None if prior_weights is None else torch.tensor(prior_weights, dtype=torch.float32),
            fixed_beta=cfg.density_path in {
                "warmstart_potential_sum_fixed_beta",
                "learnable_warmstart_potential_sum_fixed_beta",
            },
        )
        if cfg.density_path in {
            "learnable_warmstart_potential_sum",
            "learnable_warmstart_potential_sum_fixed_beta",
        }:
            return LearnableWarmStartPotentialDensityPath(
                prior=prior,
                target=target,
                target_sampleable=target,
                learnable_hiddens=list(cfg.learnable_hiddens),
                use_fourier=bool(cfg.use_fourier),
                x_fourier_dim=int(cfg.x_fourier_dim),
                x_fourier_sigma=float(cfg.x_fourier_sigma),
                t_fourier_dim=int(cfg.t_fourier_dim),
                t_fourier_sigma=float(cfg.t_fourier_sigma),
            )
        return WarmStartPotentialDensityPath(
            prior=prior,
            target=target,
            target_sampleable=target,
        )
    raise NotImplementedError(
        "Standalone repo only includes linear, learnable_linear, warmstart_potential_sum, "
        "warmstart_potential_sum_fixed_beta, learnable_warmstart_potential_sum, and "
        "learnable_warmstart_potential_sum_fixed_beta density paths."
    )
