from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.func import jacrev, jvp, vmap
from tqdm import tqdm

from src.utils.nn import FeedForward, GaussianFourierEncoder
from src.systems.base import Density, Sampleable, BumpInterval
from src.systems.density_path import DensityPath, build_density_path
from src.systems.distributions import GMM, Gaussian
from src.simulation.vector_field import VectorField, ConditionalVectorField
from src.utils.misc import get_module_device

################################
# Multi-Temperature Sampleable #
################################

class MultiTempVectorFieldWrapper(VectorField):
    """
    Obtains a regular vector field from a temperature-conditioned instance of ConditionalVectorField.
    """
    def __init__(self, vector_field: ConditionalVectorField):
        super().__init__()
        self.register_module("vector_field", vector_field)

    def drift(self, xb: Tensor, t: Tensor) -> Tensor:
        x, b = xb[:, :-1], xb[:, -1:]
        x_drift = self.vector_field(x, b, t)
        b_drift = torch.zeros_like(b)
        return torch.cat([x_drift, b_drift], dim=1)
    
class FixedTempVectorFieldWrapper(VectorField):
    """
    Obtains a regular vector field from a temperature-conditioned instance of ConditionalVectorField.
    """
    def __init__(self, vector_field: ConditionalVectorField, beta: float = 1.0):
        super().__init__()
        self.register_module("vector_field", vector_field)
        self.beta = beta

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        b = self.beta * torch.ones_like(t)
        return self.vector_field(x, b, t)


################################
# Multi-Temperature Sampleable #
################################


class MTSampleable(Sampleable):
    """
    Factored joint density over x and beta.
    """
    def __init__(self, x_sampleable: Sampleable, beta_sampleable: Sampleable):
        super().__init__()
        self.x_sampleable = x_sampleable
        self.beta_sampleable = beta_sampleable

    @property
    def dim(self) -> int:
        return self.x_sampleable.dim + self.beta_sampleable.dim

    def sample_from_betas(self, beta: Tensor) -> Tensor:
        """
        Sample at specified temperature.
        """
        x = self.x_sampleable.sample_at_temp(beta)  # (bs, dim)
        return torch.cat([x, beta], dim=1)

    def sample(self, num_samples: int) -> Tensor:
        """
        Batched sampling at arbitrary temperatures.
        Args:
            num_samples: int
            beta: (bs, 1)
        Returns:
            xb: (bs, dim + 1)
        """
        beta = self.beta_sampleable.sample(num_samples)  # (bs, 1)
        x = self.x_sampleable.sample_at_temp(beta)  # (bs, dim)
        return torch.cat([x, beta], dim=1)
    

########################################
# Temperature Reparameterization Utils #
########################################


class BetaConverter(ABC):
    """
    General framework suggested in https://journals.aps.org/pre/abstract/10.1103/PhysRevE.91.061301
    """

    @abstractmethod
    def xi_to_beta(self, xi: Tensor) -> Tensor:
        """
        Args:
            xi: (..., 1)
        Returns:
            beta: (..., 1)
        """
        pass


class PolynomialBetaConverter(BetaConverter):
    """
    Polynomial proposal from https://journals.aps.org/pre/abstract/10.1103/PhysRevE.91.061301
    """

    def __init__(self, delta: float, delta_prime: float, s: float):
        self.delta = delta
        self.delta_prime = delta_prime
        self.s = s

    def xi_to_beta(self, xi: Tensor) -> Tensor:
        """
        Args:
            xi: (..., 1)
        Returns:
            beta: (..., 1)
        """
        xi = torch.abs(xi)
        val1 = torch.tensor(0.0).to(xi)
        val2 = self.s * (
            3 * ((xi - self.delta) / (self.delta_prime - self.delta)) ** 2
            - 2 * ((xi - self.delta) / (self.delta_prime - self.delta)) ** 3
        )
        val3 = torch.tensor(self.s).to(xi)
        return 1 - torch.where(
            xi < self.delta,
            val1,
            torch.where(xi < self.delta_prime, val2, val3),
        )


def build_converter(cfg: DictConfig) -> BetaConverter:
    if cfg.converter == "polynomial":
        return PolynomialBetaConverter(
            delta=cfg.delta, delta_prime=cfg.delta_prime, s=cfg.s
        )
    else:
        raise NotImplementedError(f"Converter {cfg.converter} not implemented")
    
class XiSamplerFromBetaSampler(Sampleable):
    def __init__(
        self,
        xb_from_b: Callable,
        converter: BetaConverter,
        z_sampler: Sampleable,
    ):
        super().__init__()
        self.xb_from_b = xb_from_b
        self.converter = converter
        self.z_sampler = z_sampler

    def sample(self, num_samples: int) -> Tensor:
        z = self.z_sampler.sample(num_samples)
        b = self.converter.xi_to_beta(z)
        x = self.xb_from_b(b)[:, :-1]
        return torch.cat([x, z], dim=1)
    

#################################
# Multi-Temperature Free Energy #
#################################


class MTContinuumFreeEnergy(nn.Module, ABC):
    """
    Approximation of free energy along (t,beta) continuum.
    Evaluates to zero at t=0 so as to guarantee correctness when t=0.
    """

    @abstractmethod
    def forward(self, beta: Tensor, t: Tensor) -> Tensor:
        pass

    def free_energy_at(self, beta: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            beta: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            free_energy: (batch_size, 1)
        """
        return self(beta, t)

    def dt_free_energy_at(self, beta: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            beta: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            dt_free_energy: (batch_size, 1)
        """
        t = t.unsqueeze(1)  # (bs, 1, 1)
        beta = beta.unsqueeze(1)  # (bs, 1, 1)
        dt_free_energy = vmap(jacrev(self.free_energy_at, argnums=1))(beta, t)  # (bs, 1, 1, 1, 1)
        return dt_free_energy.reshape(-1, 1)


class ZeroMTContinuumFreeEnergy(MTContinuumFreeEnergy):
    """
    Zero free energy.
    """

    def forward(self, beta: Tensor, t: Tensor) -> Tensor:
        return torch.zeros_like(t)


class MLPMTContinuumFreeEnergy(MTContinuumFreeEnergy):
    """
    Learnable approximation of free energy along (t,beta) continuum.
    Evaluates to zero at t=0 so as to guarantee correctness when t=0.
    """

    def __init__(
            self, 
            hiddens: List[int],
            use_fourier: bool,
            b_fourier_dim: Optional[int] = None,
            b_fourier_sigma: Optional[float] = None,
            t_fourier_dim: Optional[int] = None,
            t_fourier_sigma: Optional[float] = None,
        ):
        super().__init__()
        if use_fourier:
            assert b_fourier_dim is not None and b_fourier_sigma is not None
            assert t_fourier_dim is not None and t_fourier_sigma is not None
            self.b_encoder = GaussianFourierEncoder(1, b_fourier_dim, b_fourier_sigma)
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)
            self.mlp = FeedForward([b_fourier_dim + t_fourier_dim] + hiddens + [1])
        else:
            self.mlp = FeedForward([2] + hiddens + [1])

        self.use_fourier = use_fourier

    def forward(self, b: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            beta: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            free_energy: (batch_size, 1)
        """
        if self.use_fourier:
            b_encoding = self.b_encoder(b)
            t_encoding = self.t_encoder(t)
            bt_encoding = torch.cat([b_encoding, t_encoding], dim=-1)
            output = self.mlp(bt_encoding)
        else:
            bt = torch.cat([b, t], dim=-1)
            output = self.mlp(bt)
        return t * output  # (batch_size, 1)

    def fit(self, mt_continuum: "MTContinuum", scale: float, beta_min: float, beta_max: float, grid: int=500, num_iterations: int = 2500):
        """
        Fit the free energy to a given MTContinuum.
        Args:
            mt_continuum: MTContinuum
            x_scale: float
            beta_min: float
            beta_max: float
        """
        device = get_module_device(self)

        # Part 1: Estimate the free energies
        dx = 2 * scale / grid
        nbs = 50
        nts = 50

        # (b,t) query points
        bs = torch.linspace(0.2, 1.0, nbs).to(device)
        ts = torch.linspace(0.0, 1.0, nts).to(device)
        bt = torch.stack(torch.meshgrid(bs, ts), dim=2).reshape(-1, 2).to(device) # (nbs*nts, 2)

        # (x,y) query points
        xs = torch.linspace(-scale, scale, grid).to(device)
        ys = torch.linspace(-scale, scale, grid).to(device)
        xx, yy = torch.meshgrid(xs, ys)
        xy = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device) # (grid^2, 2)

        oracle_free_energies = torch.zeros(nbs, nts).to(device)
        for b_idx, b in enumerate(bs):
            for t_idx, t in enumerate(ts):
                # Compute free energy at (b,t)
                bb = b.view(1,1).expand(xy.shape[0], 1)
                tt = t.view(1,1).expand(xy.shape[0], 1)
                xyb = torch.cat([xy, bb], dim=1) # (grid^2, 3)
                f = -torch.log(torch.sum(dx ** 2 * torch.exp((mt_continuum.log_density(xyb, t)))))
                # print(f'Free energy at (b={b.item():.2f}, t={t.item():.2f}): {f.item():.2f}')
                oracle_free_energies[b_idx, t_idx] = f



        # Part 2: Fit the free energy
        b = bt[:, 0].view(-1, 1) # (nbs*nts, 1)
        t = bt[:, 1].view(-1, 1) # (nbs*nts, 1)

        opt = torch.optim.Adam(self.parameters(), lr=1e-3)

        pbar = tqdm(range(num_iterations))
        for i in pbar:
            opt.zero_grad()
            xyb = torch.rand(1000, 3).to(device)
            ftb = self(b, t)
            loss = torch.mean((ftb - oracle_free_energies.reshape(-1,1)) ** 2)
            loss.backward()
            opt.step()
            pbar.set_description(f'Loss: {loss.item():.2f}')
                
class ReparameterizedMTContinuumFreeEnergy(nn.Module):
    def __init__(self, free_energy: MTContinuumFreeEnergy, converter: BetaConverter):
        super().__init__()
        self.register_module("free_energy", free_energy)
        self.converter = converter

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            z: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            free_energy: (batch_size, 1)
        """
        b = self.converter.xi_to_beta(z)
        return self.free_energy(b, t)
    
##########################################
# Time and Temperature Biasing Densities #
##########################################

class BiasingDensity(nn.Module, ABC):
    @abstractmethod
    def update(self, **kwargs):
        """
        Update biasing potential based on recent samples.
        """
        pass

    @abstractmethod
    def log_density(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            z: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        pass


class UniformBiasingDensity(BiasingDensity):
    """
    The default case corresponding to no biasing potential
    """

    def update(self, **kwargs):
        pass

    def log_density(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            z: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size
        """
        return torch.zeros_like(t)


def get_bin_dist(
    z: torch.Tensor, z_lower_bound: float, z_upper_bound: float, num_bins: int = 15
):
    """
    Creates a binned distribution from a given z
    Args:
    - z: (bs, 1)
    - z_lower_bound: float
    - z_upper_bound: float
    - num_bins: int
    Returns:
    - bin_dist: (nbins,)
    """
    z_bins = torch.linspace(z_lower_bound, z_upper_bound, num_bins + 1).to(z)
    z_bins = 0.5 * (z_bins[:-1] + z_bins[1:])
    z_diff = torch.abs(z - z_bins)  # (bs, nbins)
    bin_idxs = torch.argmin(z_diff, dim=-1)
    bin_counts = torch.zeros_like(z_bins)
    available_bin_idxs, available_bin_counts = torch.unique(
        bin_idxs, return_counts=True
    )
    bin_counts[available_bin_idxs] = (
        bin_counts[available_bin_idxs] + available_bin_counts
    )
    bin_dist = bin_counts / z.shape[0]
    return bin_dist


def get_mixing_metric(zs: torch.Tensor) -> torch.Tensor:
    """
    Args:
    - zs: (bs, nts, 1)
    Returns:
    - mixing_metric: ()
    """
    return torch.mean(torch.std(zs, dim=1))


def get_non_uniformity_metric(bin_dist: torch.Tensor) -> torch.Tensor:
    """
    L2 between given binned distribution and uniform (optimal) binned distribution. High non-uniformity metric indicates that the distribution is farther from uniform
    Args:
    - bin_dist: binned distribution, (nbins,)
    Returns:
    - non_uniformity_metric: ()
    """
    optimal_bin_dist = torch.ones_like(bin_dist) / bin_dist.shape[0]
    return 20 * torch.sqrt(torch.mean(torch.square(bin_dist - optimal_bin_dist)))


class GaussianKernelBiasingDensity(BiasingDensity):
    def __init__(
        self,
        t_bins: int,
        z_bins: int,
        z_lower_bound: float,
        z_upper_bound: float,
        t_std: float = 0.1,
        z_std: float = 0.1,
        update_rate: float = 0.1,
    ):
        super().__init__()
        self.z_lower_bound = z_lower_bound
        self.z_upper_bound = z_upper_bound
        self.t_std = t_std
        self.z_std = z_std
        self.update_rate = update_rate
        self.eps = 1e-2
        # Build means
        t_means = torch.linspace(0, 1, t_bins)
        z_means = torch.linspace(z_lower_bound, z_upper_bound, z_bins + 1)
        z_means = (z_means[1:] + z_means[:-1]) / 2
        zz, tt = torch.meshgrid(z_means, t_means)
        means = torch.stack([zz, tt], dim=-1)  # (nzs, nts, 2)
        weights = torch.zeros_like(means[:, :, -1:])  # (nzs, nts, 1)
        self.register_buffer("t_means", t_means)
        self.register_buffer("means", means)
        self.register_buffer("weights", weights)

    def get_update_weights(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Args:
        - z: (bs, nts, 1)
        - t: (bs, nts, 1)
        Returns:
        - weight_updates: (num_means,)
        """
        z = z.view(-1, 1)
        t = t.view(-1, 1)
        zt = torch.cat([z, t], dim=-1)  # (bs * nts, 2)
        diffs = torch.sum(
            torch.square(zt.view(-1, 1, 2) - self.means.view(-1, 2)), dim=-1
        )  # (bs * nts, num_means)
        mean_idxs = torch.argmin(diffs, dim=1)  # (bs * nts,)
        idxs, counts = torch.unique(mean_idxs, return_counts=True)
        weight_update = torch.zeros_like(self.weights.view(-1))
        weight_update[idxs] = counts / zt.shape[0]
        return weight_update.view_as(self.weights)

    def update(self, z: Tensor, t: Tensor, uniform_threshold: float = 0.4):
        """
        Args:
        - z: (bs, nts, 1)
        - t: (bs, nts, 1)
        """
        # Compute non-uniformity metric for each mean_t bin, and use these values to scale the update
        input_ts = t[0, :, 0]  # (ninput_ts,)
        mean_ts = self.t_means  # (nmean_ts,)
        mean_t_idxs = torch.argmin(
            torch.abs(input_ts.unsqueeze(-1) - mean_ts), dim=-1
        )  # (ninput_ts,)
        non_uniformity_metrics = torch.zeros_like(mean_ts)  # (nmean_ts,)
        for mean_t_idx in range(mean_ts.shape[0]):
            input_mask = mean_t_idxs == mean_t_idx  # (nmask_ts,)
            if torch.any(input_mask):
                zs_at_t = z[:, input_mask, :].view(-1, 1)  # (bs * nmask_ts,1)
                bin_dists = get_bin_dist(
                    zs_at_t, self.z_lower_bound, self.z_upper_bound
                )  # (nmask_ts, num_bins)
                non_uniformity_metrics[mean_t_idx] = get_non_uniformity_metric(
                    bin_dists
                )  # (nmask_ts,)
        if torch.max(non_uniformity_metrics) < uniform_threshold:
            return
        # (nmean_ts, 1)
        non_uniformity_metrics = non_uniformity_metrics.unsqueeze(-1)
        # Compute raw update weights
        raw_weight_update = self.get_update_weights(z, t)  # (nmean_zs, nmean_ts, 1)
        # Update weights
        self.weights = (
            self.weights + self.update_rate * raw_weight_update * non_uniformity_metrics
        )

    def log_density(self, z: Tensor, t: Tensor):
        """
        Args:
            z: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        zt = torch.cat([z, t], dim=1)  # (batch_size, 2)
        diff = torch.square(
            zt.view(-1, 1, 2) - self.means.view(1, -1, 2)
        )  # (bs, num_means, 2)
        std = torch.tensor([self.z_std, self.t_std]).to(zt)
        unweighted_potential = -0.5 * torch.sum(
            diff / (std**2), dim=-1
        )  # (bs, num_means)
        weighted_potential = unweighted_potential + torch.log(
            self.weights.view(1, -1) + self.eps
        )  # (bs, num_means)
        return -torch.sum(torch.exp(weighted_potential), dim=1, keepdim=True)


def build_biasing_density(cfg: DictConfig) -> BiasingDensity:
    if cfg.biasing_density == "zero":
        return UniformBiasingDensity()
    elif cfg.biasing_density == "gaussian":
        return GaussianKernelBiasingDensity(
            t_bins=cfg.t_bins,
            z_bins=cfg.z_bins,
            z_lower_bound=cfg.z_lower_bound,
            z_upper_bound=cfg.z_upper_bound,
            t_std=cfg.t_std,
            z_std=cfg.z_std,
            update_rate=cfg.update_rate,
        )
    else:
        raise NotImplementedError(
            f"Biasing density {cfg.biasing_density} not implemented"
        )

#######################################
# Multi-Temperature Density Continuum #
#######################################


class MTContinuum(nn.Module, ABC):
    """
    Intended as an intermediate between a density path over x, and a density path over [x, beta].
    Effectively the latter without any density over beta.
    """

    @property
    @abstractmethod
    def x_dim(self) -> int:
        pass

    @property
    def reference_start_sampleable(self) -> Sampleable:
        pass

    @property
    def reference_end_sampleable(self) -> Sampleable:
        pass

    @abstractmethod
    def sample_start_from_betas(self, betas: Tensor) -> Tensor:
        pass

    @abstractmethod
    def log_density(self, xb: Tensor, t: Tensor) -> Tensor:
        pass

    def dxb_log_density(self, xb: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xb: (batch_size, dim + 1)
            t: (batch_size, 1)
        Returns:
            dxb_log_density: (batch_size, dim + 1)
        """
        xb = xb.unsqueeze(1)  # (bs, 1, dim + 1)
        t = t.unsqueeze(1)  # (bs, 1, 1)
        dxb_log_density = vmap(jacrev(self.log_density))(xb, t)  # (bs, 1, 1, 1, dim + 1)
        return dxb_log_density.view(-1, self.x_dim + 1)

    def dt_log_density(self, xb: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xb: (batch_size, dim + 1)
            t: (batch_size, 1)
        Returns:
            dt_log_density: (batch_size, 1)
        """
        xb = xb.unsqueeze(1)  # (bs, 1, dim + 1)
        t = t.unsqueeze(1)
        dt_log_density = vmap(jacrev(self.log_density, argnums=1))(
            xb, t
        )  # (bs, 1, 1, 1, 1)
        return dt_log_density.view(-1, 1)
    
    

class MTContinuumFromPath(MTContinuum):
    """
    Construct continuum from existing density path.
    """

    def __init__(self, reference_density_path: DensityPath):
        super().__init__()
        self.register_module("reference_density_path", reference_density_path)

    @property
    def x_dim(self) -> int:
        return self.reference_density_path.dim

    @property
    def reference_start_sampleable(self) -> Sampleable:
        return self.reference_density_path.start_sampleable

    @property
    def reference_end_sampleable(self) -> Sampleable:
        return self.reference_density_path.end_sampleable

    def sample_start_from_betas(self, betas: Tensor) -> Tensor:
        x = self.reference_density_path.start_sampleable.sample_at_temp(betas)
        return torch.cat([x, betas], dim=1)

    def log_density(self, xb: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xb: (batch_size, dim + 1)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        x = xb[:, :-1]  # (batch_size, dim)
        b = xb[:, -1:]  # (batch_size, 1)
        return b * self.reference_density_path.log_density(x, t)


class MTGaussianLinearContinuum(MTContinuum):
    """
    Geometric density path from Gaussian to target.
    """

    def __init__(self, ref_start: Gaussian, ref_target: Density):
        super().__init__()
        self.register_module("ref_start", ref_start)
        self.register_module("ref_target", ref_target)

    @property
    def x_dim(self) -> int:
        return self.ref_start.dim

    @property
    def reference_start_sampleable(self) -> Sampleable:
        return self.ref_start

    @property
    def reference_end_sampleable(self) -> Sampleable:
        if isinstance(self.ref_target, Sampleable):
            return self.ref_target
        else:
            raise Exception("Reference target is not sampleable")

    def sample_start_from_betas(self, betas: Tensor) -> Tensor:
        x = self.ref_start.sample_at_temp(betas)
        return torch.cat([x, betas], dim=1)

    def log_density(self, xb: Tensor, t: Tensor) -> Tensor:
        x = xb[:, :-1]  # (batch_size, dim)
        b = xb[:, -1:]  # (batch_size, 1)
        at0 = self.ref_start.log_density_at_temp(x, b)  # (batch_size, 1)
        at1 = b * self.ref_target.log_density(x)  # (batch_size, 1)
        return (1 - t) * at0 + t * at1


class MTLearnableGaussianLinearContinuum(MTContinuum):
    """
    Geometric density path from Gaussian to target.
    """

    def __init__(
        self, ref_start: Gaussian, ref_target: Density, learnable_hiddens: List[int]
    ):
        super().__init__()
        self.register_module("ref_start", ref_start)
        self.register_module("ref_target", ref_target)
        self.learnable_potential = FeedForward([self.x_dim + 1] + learnable_hiddens + [1])

    @property
    def x_dim(self) -> int:
        return self.ref_start.dim

    @property
    def reference_start_sampleable(self) -> Sampleable:
        return self.ref_start

    @property
    def reference_end_sampleable(self) -> Sampleable:
        if isinstance(self.ref_target, Sampleable):
            return self.ref_target
        else:
            raise Exception("Reference target is not sampleable")

    def sample_start_from_betas(self, betas: Tensor) -> Tensor:
        x = self.ref_start.sample_at_temp(betas)
        return torch.cat([x, betas], dim=1)

    def log_density(self, xb: Tensor, t: Tensor) -> Tensor:
        x = xb[:, :-1]  # (batch_size, x_dim)
        xt = torch.cat([x,t], dim=1) # (batch_size, x_dim + 1)
        b = xb[:, -1:]  # (batch_size, 1)
        at0 = self.ref_start.log_density_at_temp(x, b)  # (batch_size, 1)
        at1 = b * self.ref_target.log_density(x)  # (batch_size, 1)
        return (1 - t) * at0 + t * at1 + b * t * (1 - t) * self.learnable_potential(xt)


class LinearGaussianContinuum(MTContinuum):
    """
    MTContinuum analogue of LinearGaussianInterpolationPath.
    """

    def __init__(self, ref_start: Gaussian, ref_end: Gaussian):
        super().__init__()
        self.register_module("ref_start", ref_start)
        self.register_module("ref_end", ref_end)

    @property
    def x_dim(self) -> int:
        return self.ref_start.dim

    @property
    def reference_start_sampleable(self) -> Sampleable:
        return self.ref_start

    def sample_start_from_betas(self, betas: Tensor) -> Tensor:
        x = self.ref_start.sample_at_temp(betas)
        return torch.cat([x, betas], dim=1)

    def mu_0(self, beta: Tensor) -> Tensor:
        return self.ref_start.mean.unsqueeze(0).expand(beta.shape[0], self.x_dim)

    def mu_1(self, beta: Tensor) -> Tensor:
        return self.ref_end.mean.unsqueeze(0).expand(beta.shape[0], self.x_dim)

    def sigma_0(self, beta: Tensor) -> Tensor:
        return self.ref_start.cov.unsqueeze(0).expand(
            beta.shape[0], self.x_dim, self.x_dim
        ) / beta.unsqueeze(-1)

    def sigma_1(self, beta: Tensor) -> Tensor:
        """
        Args:
            beta: (batch_size, 1)
        Returns:
            sigma_1: (batch_size, dim, dim)
        """
        return self.ref_end.cov.unsqueeze(0).expand(
            beta.shape[0], self.x_dim, self.x_dim
        ) / beta.unsqueeze(-1)

    def get_mu_t(self, beta: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            beta: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            mu_t: (batch_size, dim)
        """
        mu_0 = self.mu_0(beta)  # (batch_size, dim)
        mu_1 = self.mu_1(beta)  # (batch_size, dim)
        return (1 - t) * mu_0 + t * mu_1

    def get_sigma_t(self, beta: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            beta: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            sigma_t: (batch_size, dim, dim)
        """
        sigma_0 = self.sigma_0(beta)  # (batch_size, dim, dim)
        sigma_1 = self.sigma_1(beta)  # (batch_size, dim, dim)
        t = t.unsqueeze(-1)
        return (1 - t) * sigma_0 + t * sigma_1

    def get_sigma_t_chol(self, beta: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            beta: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            sigma_t_chol: (batch_size, dim, dim)
        """
        return torch.linalg.cholesky(self.get_sigma_t(beta, t))

    def dt_mu_t(self, beta: Tensor) -> Tensor:
        """
        Args:
            beta: (batch_size, 1)
        Returns:
            dt_mu_t: (batch_size, dim)
        """
        return self.mu_1(beta) - self.mu_0(beta)

    def log_density(self, xb: Tensor, t: Tensor) -> Tensor:
        x, b = xb[:, :-1], xb[:, -1:]
        mu_t = self.get_mu_t(b, t)  # (batch_size, dim)
        sigma_t = self.get_sigma_t(b, t)
        p_t = MultivariateNormal(mu_t, sigma_t, validate_args=False)
        return p_t.log_prob(x).unsqueeze(-1)


class LinearGaussianContinuumVectorField(ConditionalVectorField):
    def __init__(self, mt_continuum: LinearGaussianContinuum):
        super().__init__()
        self.mt_continuum = mt_continuum

    def drift(self, x: Tensor, b: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xb: (batch_size, dim + 1)
            t: (batch_size, 1)
        Returns:
            drift: (batch_size, dim)
        """
        mu_t = self.mt_continuum.get_mu_t(b, t)  # (batch_size, dim)
        sigma_t_cholesky = self.mt_continuum.get_sigma_t_chol(
            b, t
        )  # (batch_size, dim, dim)
        A = torch.bmm(
            torch.linalg.inv(sigma_t_cholesky), (x - mu_t).unsqueeze(-1)
        )  # (batch_size, dim, 1)
        dt_mu_t = self.mt_continuum.dt_mu_t(b)  # (batch_size, dim)
        dt_sigma_t_cholesky = vmap(
            jacrev(self.mt_continuum.get_sigma_t_chol, argnums=1)
        )(
            b.unsqueeze(1), t.unsqueeze(1)
        )  # (batch_size, 1, dim, dim, 1, 1)
        dt_sigma_t_cholesky = dt_sigma_t_cholesky.squeeze(
            1, 4, 5
        )  # (batch_size, dim, dim)
        return torch.bmm(dt_sigma_t_cholesky, A).squeeze(-1) + dt_mu_t


def build_mt_continuum(cfg: DictConfig) -> MTContinuum:
    if cfg.mt_continuum == "from_density_path":
        density_path = build_density_path(cfg)
        return MTContinuumFromPath(reference_density_path=density_path)
    elif cfg.mt_continuum == "linear":
        # Build source and target and construct continuum
        source = Gaussian.isotropic(cfg.x_dim, cfg.source_std)
        if cfg.target == "fab_gmm":
            target = GMM.FAB_GMM(cov_scale=cfg.cov_scale)
        else:
            raise NotImplementedError(f"Target {cfg.target} not implemented")
        return MTGaussianLinearContinuum(ref_start=source, ref_target=target)
    elif cfg.mt_continuum == "learnable_linear":
        source = Gaussian.isotropic(cfg.x_dim, cfg.source_std)
        if cfg.target == "fab_gmm":
            target = GMM.FAB_GMM(cov_scale=cfg.cov_scale)
        else:
            raise NotImplementedError(f"Target {cfg.target} not implemented")
        return MTLearnableGaussianLinearContinuum(
            ref_start=source, ref_target=target, learnable_hiddens=cfg.learnable_hiddens
        )
    else:
        raise NotImplementedError(f"MTContinuum {cfg.mt_continuum} not implemented")

class ReparameterizedMTContinuum(nn.Module, ABC):
    """
    Re-parameterization via beta = f(xi), implemented as wrapper around MTContinuum. Intended for use in temperature-mixed dynamics, and therofore also includes a biasing potential over xi.
    """

    @property
    @abstractmethod
    def x_dim(self) -> int:
        """Returns the state dim of the system (i.e., dim of x)."""
        pass 

    @property
    @abstractmethod
    def source_sampleable(self) -> Sampleable:
        pass

    @abstractmethod
    def log_density(self, xz: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xz: (batch_size, dim + 1)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        pass
    
    def dt_log_density(self, xz: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xz: (batch_size, dim + 1)
            t: (batch_size, 1)
        Returns:
            dt_log_density: (batch_size, 1)
        """
        xz = xz.unsqueeze(1)  # (bs, 1, dim + 1)
        t = t.unsqueeze(1)  # (bs, 1, 1)
        dt_log_density = vmap(jacrev(self.log_density, argnums=1))(xz, t)  # (bs, 1, 1, 1, 1)
        return dt_log_density.view(-1, 1)

    def dxz_log_density(self, xz: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xz: (batch_size, dim + 1)
            t: (batch_size, 1)
        Returns:
            dxz_log_density: (batch_size, dim)
        """
        xz = xz.unsqueeze(1)  # (bs, 1, dim + 1)
        t = t.unsqueeze(1)  # (bs, 1, 1)
        dxz_log_density = vmap(jacrev(self.log_density))(xz, t)  # (bs, 1, 1, 1, x_dim + 1)
        return dxz_log_density.view(-1, xz.shape[-1])

class WrapperReparameterizedMTContinuum(ReparameterizedMTContinuum):
    """
    Re-parameterization via beta = f(xi), implemented as wrapper around MTContinuum. Intended for use in temperature-mixed dynamics, and therofore also includes a biasing potential over xi.
    """

    def __init__(
        self,
        mt_continuum: MTContinuum,
        z_source: Sampleable,
        converter: BetaConverter,
        z_confining_density_path: DensityPath,
        bt_free_energy: MTContinuumFreeEnergy,
        z_biasing_density: Optional[BiasingDensity],
    ):
        """
        Args:
            mt_continuum: continuum over beta
            converter: re-parameterization expressing beta as a function of xi
        Constraints:
        - xi_source must correspond to the density given by the combination of mt_continuum, confining density, metadynamics density, and free energy.
        """
        super().__init__()
        self.register_module("mt_continuum", mt_continuum)
        self.register_module("z_source", z_source)
        self.register_module("z_confining_density_path", z_confining_density_path)
        if z_biasing_density is None:
            z_biasing_density = UniformBiasingDensity()
        self.register_module("z_biasing_density", z_biasing_density)
        self.register_module("free_energy", bt_free_energy)
        self.converter = converter

    @property
    def x_dim(self) -> int:
        """Returns the state dim of the system (i.e., dim of x)."""
        return self.mt_continuum.x_dim

    @property
    def source_sampleable(self) -> Sampleable:
        return XiSamplerFromBetaSampler(
            xb_from_b=self.mt_continuum.sample_start_from_betas,
            converter=self.converter,
            z_sampler=self.z_source,
        )

    def sample_start(self, z: Tensor) -> Tensor:
        """
        Args:
            z: (batch_size, 1)
        Returns:
            xz: (batch_size, dim + 1)
        """
        b = self.converter.xi_to_beta(z)
        x = self.mt_continuum.sample_start_from_betas(b)[:, :-1]
        xz = torch.cat([x, z], dim=1)
        return xz

    def log_density(self, xz: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            xz: (batch_size, dim + 1)
            t: (batch_size, 1)
        Returns:
            log_density: (batch_size, 1)
        """
        x = xz[:, :-1]
        z = xz[:, -1:]
        b = self.converter.xi_to_beta(z)
        xb = torch.cat([x, b], dim=1)
        return (
            self.mt_continuum.log_density(xb, t)
            + self.z_biasing_density.log_density(z, t)
            + self.z_confining_density_path.log_density(z, t)
            + self.free_energy(b, t)
        )
    

class ReparameterizedConditionalVectorField(ConditionalVectorField):
    """
    Reparameterization from a beta-dependent vector field to a xi-dependent vector field
    """
    def __init__(self, vector_field: ConditionalVectorField, converter: BetaConverter):
        super().__init__()
        self.vector_field = vector_field
        self.converter = converter

    def drift(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, dim)
            z: (batch_size, 1)
            t: (batch_size, 1)
        Returns:
            drift: (batch_size, dim)
        """
        b = self.converter.xi_to_beta(z)
        return self.vector_field(x, b, t)