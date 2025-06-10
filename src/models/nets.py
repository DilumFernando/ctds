import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule
from torch import Tensor
import wandb

from src.datasets import DummyDataloader
from src.models.utils import AnnealingScheduler
from src.utils.nn import FeedForward, GaussianFourierEncoder
from src.simulation.vector_field import VectorField, MLPVectorField, ZeroVectorField
from src.simulation.dynamics import (
    AuxiliaryProcess,
    DivegenceAuxiliaryProcess,
    ODEProcess,
)
from src.simulation.langevin import (
    AnnealedOverdampedLangevin,
    AnnealedUnderdampedLangevin,
    build_annealed_hamiltonian
)
from src.systems.base import Density
from src.systems.density_path import DensityPath, build_density_path
from src.systems.distributions import Sampleable
from src.utils.metrics import MAX_LOG_WEIGHT, ess, w2_from_samples
from src.utils.misc import cuda_profile, get_module_device, record_every_idxs, BYTES_PER_GIB


class DtFreeEnergy(nn.Module, ABC):
    @abstractmethod
    def forward(self, time: Tensor) -> Tensor:
        pass


class ZeroDtFreeEnergy(DtFreeEnergy):
    """
    Special case of a normalized density path where the free energy is constant.
    """

    @cuda_profile
    def forward(self, time: Tensor) -> Tensor:
        """
        Args:
            t: (batch_size, 1)
        Returns:
            dt_free_energy: (batch_size, 1)
        """
        return torch.zeros_like(time)


class MLPDtFreeEnergy(DtFreeEnergy):
    """Approximates the derivative d/dt(-log Z_t) of free energy with respect to time using Fourier features."""

    def __init__(
        self,
        hiddens: List[int],
        use_fourier: bool,
        t_fourier_dim: Optional[int] = None,
        t_fourier_sigma: Optional[float] = None,
    ):
        super().__init__()
        if use_fourier:
            assert t_fourier_dim is not None and t_fourier_sigma is not None
            mlp = FeedForward([t_fourier_dim] + hiddens + [1])
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)
            self.net = nn.Sequential(self.t_encoder, mlp)
        else:
            self.net = FeedForward([1] + hiddens + [1])

    @cuda_profile
    def forward(self, time: Tensor) -> Tensor:
        """
        Args:
            time: (batch_size, 1)
        Returns:
            dt_free_energy: (batch_size, 1)
        """
        return self.net(time)


#################################
# Jarzynski Auxiliary Processes #
#################################


class OverdampedJarzynski(AuxiliaryProcess):
    """
    Computes the Jarzynski importance weights of overdamped Langevin dynamics
    See https://arxiv.org/abs/2410.02711
    """

    def __init__(self, control: VectorField, density_path: DensityPath, divergence_mode: str):
        super().__init__()
        self.control = control
        self.density_path = density_path
        self.divergence_mode = divergence_mode

    def initial_value(self, num_samples: int) -> Tensor:
        """
        Initial value for the auxiliary process
        Args:
        - num_samples: int
        Returns:
        - a0: (num_samples, 1)
        """
        return torch.zeros(num_samples, 1)

    def integrate_step(self, at: Tensor, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
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
        div = self.control.divergence(xt, t, self.divergence_mode)  # (batch_size, 1)

        dx_log_pt = self.density_path.dx_log_density(xt, t)  # (batch_size, dim)
        control = self.control(xt, t)  # (batch_size, dim)
        dot = (dx_log_pt * control).sum(dim=-1, keepdim=True)  # (batch_size, 1)

        dt_log_pt = self.density_path.dt_log_density(xt, t)  # (batch_size, 1)

        return at + (div + dot + dt_log_pt) * dt


class UnderdampedJarzynski(AuxiliaryProcess):
    """
    Computes the Jarzynski importance weights of underdamped Langevin dynamics
    """

    def __init__(self, control: VectorField, density_path: DensityPath, divergence_mode: str):
        super().__init__()
        self.control = control
        self.density_path = density_path
        self.x_dim = density_path.dim
        self.divergence_mode = divergence_mode

    def initial_value(self, num_samples: int) -> Tensor:
        """
        Initial value for the auxiliary process
        Args:
        - num_samples: int
        Returns:
        - a0: (num_samples, 1)
        """
        return torch.zeros(num_samples, 1)

    def integrate_step(self, at: Tensor, x_px: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        """
        Integration step for the Jarzynski importance weights
        Args:
        - at: (batch_size, 1)
        - x_px: (batch_size, 2 * dim)
        - t: (batch_size, 1)
        - dt: (batch_size, 1)
        Returns:
        - updated_at: (batch_size, 1)
        """
        x = x_px[:, : self.x_dim]
        div = self.control.divergence(x, t, self.divergence_mode)  # (batch_size, 1)

        dx_log_pt = self.density_path.dx_log_density(x, t)  # (batch_size, dim)
        control = self.control(x, t)  # (batch_size, dim)
        dot = (dx_log_pt * control).sum(dim=-1, keepdim=True)  # (batch_size, 1)

        dt_log_pt = self.density_path.dt_log_density(x, t)  # (batch_size, 1)

        return at + (div + dot + dt_log_pt) * dt


###################
# Model Proposals #
###################


class PINNProposal(nn.Module, ABC):
    @abstractmethod
    def sample(self, ts: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
        - ts: (batch_size, num_integration_steps + 1, 1)
        Returns:
        - xs: (batch_size, num_integration_steps + 1, data_dim)
        - ts: (batch_size, num_integration_steps + 1, 1)
        - log_weights: (batch_size, num_integration_steps + 1, 1)
        """
        raise NotImplementedError
    
    def get_metrics(self, ts: torch.Tensor, label: str) -> Dict:
        """
        Returns proposal-specific metrics to log to e.g., WandB
        Args:
        - ts: (bs, nts, 1)
        - label: str, prefix label for the metrics
        """
        return {}

class ODEProposal(PINNProposal):
    def __init__(
        self,
        control: VectorField,
        source_sampleable: Sampleable,
        record_every: int,
        divergence_mode: str,
        target_sampleable: Optional[Sampleable] = None,

    ):
        super().__init__()
        self.auxiliary_processes = {
            'cov': DivegenceAuxiliaryProcess(control, divergence_mode=divergence_mode)
        }
        self.source_sampleable = source_sampleable
        self.dynamics = ODEProcess(
            source=source_sampleable,
            vector_field=control, 
            auxiliary_processes=self.auxiliary_processes
        )
        self.record_every = record_every
        self.x_dim = source_sampleable.dim
        self.has_density = isinstance(source_sampleable, Density)

        # Store these for computing metrics
        self.control = control
        self.divergence_mode = divergence_mode
        self.target_sampleable = target_sampleable

    def sample(self, ts, **kwargs):
        batch_size = ts.shape[0]
        trajectory, aux_trajectory, ts = self.dynamics.sample_with_trajectory(
            ts=ts, num_samples=batch_size, record_every=self.record_every, **kwargs
        )


        # Compute weights
        if self.has_density:
            # Compute change-of-variables formula
            source = self.dynamics.source
            x0 = trajectory[:, 0, : self.x_dim]  # (batch_size, data_dim)
            init_log_weights = source.log_density(x0).view(-1,1,1) # (batch_size, 1, 1)
            d_log_weights = aux_trajectory["cov"]  # (batch_size, num_integration_steps + 1, 1)
            log_weights = init_log_weights + d_log_weights  # (batch_size, num_integration_steps + 1, 1)
        else:
            log_weights = torch.zeros_like(ts)

        return trajectory, ts, log_weights
    
    def get_metrics(self, ts: torch.Tensor, label: str, **kwargs) -> Dict:
        """
        Returns proposal-specific metrics to log to e.g., WandB
        Args:
        - ts: (bs, nts, 1)
        - label: str, prefix label for the metrics
        """
        # Compute weights
        trajectory, ts, log_weights = self.sample(ts, **kwargs)

        metrics = {}

        model_samples = trajectory[:, -1, : self.x_dim]  # (batch_size, data_dim)

        # Compute ELBO
        # TODO: Come up with better variable name rather than re-using the below
        log_model_weights = log_weights[:, -1, :]  # (batch_size, 1)
        log_target_weights = self.target_sampleable.log_density(model_samples)  # (batch_size, 1)
        elbo = torch.mean(log_target_weights - log_model_weights).item()
        metrics[f"{label}_elbo"] = elbo

        # Compute W2, R-ESS, EUBO if end_sampleable is provided
        if self.target_sampleable is not None:
            target_samples = self.target_sampleable.sample(model_samples.shape[0]) # (batch_size, data_dim)
            w2_val = w2_from_samples(model_samples, target_samples)
            metrics[f"{label}_w2"] = w2_val.item()

            # Get compute weights under model ODE
            reverse_ts = ts.flip(1)  # (batch_size, num_integration_steps + 1, 1)
            x0, aux_dict = self.dynamics.sample(
                ts=reverse_ts, x0=target_samples
            )
            log_model_weights = self.source_sampleable.log_density(x0) - aux_dict["cov"]
            log_target_weights = self.target_sampleable.log_density(target_samples) # (batch_size, 1)
            
            # Compute R-ESS
            ess_log_weights = log_model_weights - log_target_weights # (batch_size, 1)
            r_ess = ess(ess_log_weights).item()
            metrics[f"{label}_r_ess"] = r_ess

            # Compute EUBO
            eubo = torch.mean(log_target_weights - log_model_weights).item()
            metrics[f"{label}_eubo"] = eubo

        return metrics

class LangevinProposal(PINNProposal):
    """
    Model proposal based on Langevin dynamics, optionally reweighted with Jarzynski importance weights.
    """

    def sample(self, ts: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        # Choose random timesteps to integrate against
        batch_size = ts.shape[0]
        trajectory, aux_trajectory, ts = self.dynamics.sample_with_trajectory(
            ts=ts, num_samples=batch_size, record_every=self.record_every, use_tqdm=True, **kwargs
        )  # (bs, nts, data_dim), (bs, nts, 1), (bs, nts, 1)
        trajectory = trajectory[:, :, : self.x_dim]
        if self.use_jarzynski:
            log_weights = aux_trajectory[
                "jarzynski"
            ]  # (batch_size, num_integration_steps + 1, 1)
        else:
            log_weights = torch.zeros_like(ts)
        return trajectory, ts, log_weights


class OverdampedLangevinProposal(LangevinProposal):
    def __init__(
        self,
        control: VectorField,
        damping: float,
        density_path: DensityPath,
        divergence_mode: str,
        record_every: int = 1,
        use_jarzynski: bool = True,
    ):
        super().__init__()
        jarzynski_dynamics = OverdampedJarzynski(
            control=control, density_path=density_path, divergence_mode=divergence_mode
        )
        aux_processes = {"jarzynski": jarzynski_dynamics} if use_jarzynski else {}
        self.dynamics = AnnealedOverdampedLangevin(
            control=control,
            damping=damping,
            density_path=density_path,
            auxiliary_processes=aux_processes,
        )
        self.use_jarzynski = use_jarzynski
        self.record_every = record_every
        self.x_dim = density_path.dim


class UnderdampedLangevinProposal(LangevinProposal):
    def __init__(
        self,
        control: VectorField,
        scale: float,
        damping: float,
        mass: float,
        density_path: DensityPath,
        hamiltonian_type: str,
        divergence_mode: str,
        record_every: int = 1,
        use_jarzynski: bool = True,
    ):
        super().__init__()
        jarzynski_dynamics = UnderdampedJarzynski(
            control=control, density_path=density_path, divergence_mode=divergence_mode
        )
        aux_processes = {"jarzynski": jarzynski_dynamics} if use_jarzynski else {}
        hamiltonian = build_annealed_hamiltonian(
            density_path=density_path,
            hamiltonian_type=hamiltonian_type,
            mass=mass,
        )
        self.dynamics = AnnealedUnderdampedLangevin(
            control=control,
            scaling=scale,
            damping=damping,
            hamiltonian=hamiltonian,
            auxiliary_processes=aux_processes,
        )
        self.use_jarzynski = use_jarzynski
        self.record_every = record_every
        self.x_dim = density_path.dim


class UniformPINNProposal(PINNProposal):
    def __init__(self, x_dim: int, ts: Tensor, scales: Tensor, record_every: int = 1):
        """
        Args:
        - x_dim: int, dimension of the data
        - ts: (nts,)
        - scales: (nts,)
        """
        super().__init__()
        self.x_dim = x_dim
        self.ts = ts
        self.scales = scales
        self.record_every = record_every

    def sample(self, ts: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample x-coordinates uniformly from [-scale, scale] ^ data_dim, and intended mostly as a debugging tool for low-dimensional systems, but is used in e.g., https://arxiv.org/abs/2407.07873
        Args
        - ts: (batch_size, num_integration_steps + 1, 1)
        Returns:
        - xs: (batch_size, num_integration_steps + 1, data_dim)
        - ts: (batch_size, num_integration_steps + 1, 1)
        - log_weights: (batch_size, num_integration_steps + 1, 1)
        """
        # Process self.record_every
        num_ts = ts.shape[1]
        idxs = record_every_idxs(num_ts, self.record_every)
        ts = ts.clone()[:, idxs, :]  # (batch_size, num_integration_steps + 1, 1)

        t_diffs = torch.abs(
            ts - self.ts
        )  # (batch_size, num_integration_steps + 1, nts)
        closest_ts = torch.argmin(
            t_diffs, dim=-1, keepdim=True
        )  # (batch_size, num_integration_steps + 1, 1)
        # (batch_size, num_integration_steps + 1, 1)
        scales = self.scales[closest_ts]
        bs, num_ts, _ = ts.shape
        xs = scales * (2 * torch.rand(bs, num_ts, self.x_dim) - 1).to(
            ts
        )  # (bs, num_integration_steps, data_dim)
        return xs, ts, torch.zeros_like(ts)

class NETSModule(LightningModule):
    """
    General framework for implementing PINN-based training procedures including
        1. Non-Equilibrium Transport Sampler (https://arxiv.org/abs/2410.02711)
        2. Dynamical Measure Transport and Neural PDE Solvers for Sampling (https://arxiv.org/abs/2407.07873)
        3. Learning Interpolations between Boltzmann Densities (https://arxiv.org/abs/2407.07873)
    """

    def __init__(
        self,
        cfg: DictConfig,
        density_path: Optional[DensityPath] = None,
        control: Optional[VectorField] = None,
        dt_free_energy: Optional[DtFreeEnergy] = None,
    ):
        super().__init__()
        # Non-cfg hyperparameters only passed in during non-training context
        self.save_hyperparameters(ignore=["density_path", "control", "dt_free_energy"])
        self.cfg = cfg
        self.x_dim = cfg.x_dim

        # Dynamics
        if density_path is None:
            self.density_path = build_density_path(cfg)
        else:
            self.density_path = density_path

        if control is not None:
            self.control = control
        elif cfg.control == "mlp":
            if self.cfg.use_fourier:
                self.control = MLPVectorField(
                    data_dim=self.x_dim,
                    hidden_dims=cfg.control_hiddens,
                    use_fourier=True,
                    x_fourier_dim=cfg.x_fourier_dim,
                    x_fourier_sigma=cfg.x_fourier_sigma,
                    t_fourier_dim=cfg.t_fourier_dim,
                    t_fourier_sigma=cfg.t_fourier_sigma,
                )
            else:
                self.control = MLPVectorField(
                    data_dim=self.x_dim,
                    hidden_dims=cfg.control_hiddens,
                    use_fourier=False,
                )
        elif cfg.control == "zero":
            self.control = ZeroVectorField(out_dim=self.x_dim)
        else:
            raise ValueError("Invalid drift type")

        if dt_free_energy is not None:
            self.dt_free_energy = dt_free_energy
        elif cfg.free_energy == "mlp":
            if self.cfg.use_fourier:
                self.dt_free_energy = MLPDtFreeEnergy(
                    hiddens = cfg.free_energy_hiddens,
                    use_fourier = cfg.use_fourier,
                    t_fourier_dim = cfg.t_fourier_dim,
                    t_fourier_sigma = cfg.t_fourier_sigma,
                )
            else:
                self.dt_free_energy = MLPDtFreeEnergy(
                    hiddens = cfg.free_energy_hiddens, 
                    use_fourier = cfg.use_fourier
                )
        elif cfg.free_energy == "zero":
            self.dt_free_energy = ZeroDtFreeEnergy()
        else:
            raise ValueError("Invalid free energy type")

        # For backwards compatibility with older configs
        if "divergence_mode" not in cfg:
            with open_dict(cfg):
                try:
                    cfg.divergence_mode = cfg.divergence.mode
                except:
                    raise AttributeError("Please specify cfg.divergence.mode in the config!")

        # PINN-loss args
        self.annealing_scheduler = AnnealingScheduler.build(cfg)
        self.integration_avg_dt = (
            cfg.avg_dt
        )  # Average because we randomize the timesteps
        self.proposal_type = cfg.proposal

    @property
    def T(self) -> float:
        return self.annealing_scheduler.T

    def move_to_device(self, device: torch.device, verbose=False):
        """
        For use switching devices when working in e.g., a Jupyter notebook.
        """
        for name, child in self.named_children():
            child.to(device)
            if verbose:
                print(f"Moving {name} to {self.device}")
        return self

    def setup(self, stage):
        # Move all submodules to device
        for child in self.children():
            child.to(self.device)
        # Initialize losses
        self.train_losses = []
        self.train_memory_usages = []
        self.val_losses = []

    def pinn_loss(self, x: Tensor, t: Tensor, weights: Tensor):
        """
        Compute the PINN loss up to time T given fixed  samples
        Note: we are pretending here that our integration steps are uniform (they are only approximately uniform)
        Args:
            x: (batch_size, data_dim)
            t: (batch_size, 1)
            weights: (batch_size, 1)
        Returns:
            pinn_loss: (1)
        """
        # \partial_t F_t
        dt_Ft = self.dt_free_energy(
            t, profile=self.cfg.memory_profile
        )  # (batch_size, 1)

        # control(x,t) \dot \nabla_x ln p_t(xt)
        control = self.control(x, t)  # (batch_size, data_dim)
        score = self.density_path.dx_log_density(
            x, t, profile=self.cfg.memory_profile
        )  # (batch_size, data_dim)
        dot = (control * score).sum(-1, keepdims=True)  # (batch_size, 1)

        # \nabla \cdot control(x,t)
        div = self.control.divergence(
            x, t, self.cfg.divergence_mode, profile=self.cfg.memory_profile
        )  # (batch_size, 1)

        # \partial_t \ln p(x,t)
        dt_ln_pt = self.density_path.dt_log_density(
            x, t, profile=self.cfg.memory_profile
        )  # (batch_size, 1)

        # Combine to compute PINN loss
        raw_loss = (dt_Ft + div + dot + dt_ln_pt) ** 2  # (batch_size, 1)
        # TODO: reevaluate usage of mean over all samples, instead of just over timesteps...
        return torch.mean(weights * raw_loss)

    def get_integration_ts(self, T: float) -> Tensor:
        """
        Construct integration steps by which to discretize PINN integral
        Args:
        - T: float, upper bound of integration
        Returns:
        - ts: (nts, 1)
        """
        num_integration_steps = math.ceil(T / self.integration_avg_dt)
        integration_ts = T * torch.sort(torch.rand(num_integration_steps - 1)).values
        integration_ts = torch.cat([torch.zeros(1), integration_ts, torch.ones(1) * T])
        return integration_ts.unsqueeze(-1)

    def build_proposal(self, proposal_type: str) -> PINNProposal:
        if proposal_type == "overdamped_langevin":
            return OverdampedLangevinProposal(
                control=self.control,
                damping=self.cfg.damping,
                density_path=self.density_path,
                record_every=self.cfg.record_every,
                use_jarzynski=self.cfg.use_jarzynski,
                divergence_mode=self.cfg.divergence_mode,
            )
        elif proposal_type == "underdamped_langevin":
            return UnderdampedLangevinProposal(
                control=self.control,
                scale=self.cfg.scale,
                damping=self.cfg.damping,
                mass=self.cfg.mass,
                density_path=self.density_path,
                hamiltonian_type=self.cfg.hamiltonian_type,
                record_every=self.cfg.record_every,
                use_jarzynski=self.cfg.use_jarzynski,
                divergence_mode=self.cfg.divergence_mode,
            )
        elif proposal_type == "ode":
            # Note: refactor hack below
            has_target_sampleable = True
            try:
                _ = self.density_path.end_sampleable
            except NotImplementedError:
                has_target_sampleable = False

            return ODEProposal(
                control=self.control,
                source_sampleable=self.density_path.start_sampleable,
                divergence_mode=self.cfg.divergence_mode,
                record_every=self.cfg.record_every,
                target_sampleable=self.density_path.end_sampleable if has_target_sampleable else None,
            )
        elif proposal_type == "uniform":
            return UniformPINNProposal(
                x_dim=self.cfg.x_dim,
                scales=torch.Tensor(self.cfg.uniform_proposal_scales).to(
                    get_module_device(self)
                ),
                ts=torch.Tensor(self.cfg.uniform_proposal_times).to(
                    get_module_device(self)
                ),
                record_every=self.cfg.record_every,
            )
        else:
            raise ValueError("Invalid proposal type")

    def extract_weights(self, log_weights: Tensor) -> Tensor:
        # Clamp weights to avoid numerical instability
        log_weights = torch.clamp(
            log_weights, max=MAX_LOG_WEIGHT
        )  # (bs, num_integration_steps, 1)
        # Exponentiate to get weights
        weights = torch.exp(log_weights)  # (bs, num_integration_steps, 1)
        # Batch-size-aware normalization (divide out normalized weights by batch size)
        weights = weights / torch.mean(
            weights, dim=0, keepdim=True
        )  # (bs, num_integration_steps, 1)
        return weights
    
    def replenish_sample_buffer(self, num_trajectories: int, proposal_type: str, integration_ts: Optional[Tensor] = None, T: float = 1.0) -> Dict:
        """
        Grab samples with which to replenish a sample buffer
        """
        # Construct integration steps by which to discretize PINN integral
        if integration_ts is None:
            integration_ts = self.get_integration_ts(T).to(get_module_device(self))
        integration_ts = integration_ts.unsqueeze(0).expand(
            num_trajectories, -1, 1
        )  # (bs, nts, 1)

        proposal = self.build_proposal(proposal_type).to(get_module_device(self))
        xs, ts, log_weights = proposal.sample(
            ts=integration_ts
        )  # (bs, nts, data_dim), (bs, nts, 1), (bs, nts, 1)

        # Exponentiate and normalize importance weights
        weights = self.extract_weights(log_weights)

        return {
            "xs": xs.detach(), # (bs, nts, data_dim)
            "ts": ts.detach(), # (bs, nts, 1)
            "log_weights": log_weights.detach(),  # (bs, nts, 1)
            "weights": weights.detach(), # (bs, nts, 1)
        }


    def pinn_loss_wrapper(
        self,
        num_trajectories: int,
        sample_buffer: Optional[Dict] = None,
        proposal_type: Optional[str] = None,
        integration_ts: Optional[Tensor] = None,
        T: float = 1.0,
    ):
        """
        Compute PINN loss, reconstructing sample buffer if necessary
        Args:
        - batch_size: number of samples which to evaluate the PINN loss (used to subsample from the sample buffer)
        - sample_buffer: Optional[Dict], pre-computed sample buffer to use for sampling
        - num_trajectories: int, number of trajectories to sample (each corresponding to num_timesteps + 1 samples)
        - proposal_type: str, type of proposal to use (e.g, "overdamped_langevin", "underdamped_langevin", "ode")
        - integration_ts: (nts, 1), time steps to integrate against
        - T: float, upper bound of integration
        """
        if sample_buffer:
            # Using persistent buffer - need to subsample
            subsample_idxs = torch.randperm(sample_buffer["xs"].shape[0])[:num_trajectories]
            xs = sample_buffer["xs"][subsample_idxs]  # (bs, nts, data_dim)
            ts = sample_buffer["ts"][subsample_idxs] # (bs, nts, 1)
            weights = sample_buffer["weights"][subsample_idxs]  # (bs, nts, 1)
        else:
            # No persistent buffer
            assert num_trajectories is not None and proposal_type is not None
            sample_buffer = self.replenish_sample_buffer(
                num_trajectories=num_trajectories,
                proposal_type=proposal_type,
                integration_ts=integration_ts,
                T=T,
            )
            xs = sample_buffer["xs"]  # (bs, nts, data_dim)
            ts = sample_buffer["ts"]  # (bs, nts, 1)
            weights = sample_buffer["weights"]  # (bs, nts, 1)

        # Flatten xs, ts, weights
        xs = xs.reshape(-1, self.x_dim)  # (bs * nts, data_dim)
        ts = ts.reshape(-1, 1)  # (bs * nts, 1)
        weights = weights.reshape(-1, 1)  # (bs * nts, 1)
  
        # Detach samples to avoid backpropagating through the dynamics
        return self.pinn_loss(xs.detach().clone(), ts.detach().clone(), weights.detach().clone())

    def on_train_start(self):
        if self.cfg.wandb:
            # Log config
            wandb.config.update(OmegaConf.to_container(self.cfg, resolve=True))
            # Monitor gradients and parameters
            wandb.watch(self, log="all", log_freq=self.cfg.train_steps_per_epoch)  # log="all" logs gradients & parameters
    
    def on_train_epoch_start(self):
        """
        Replenish global sample buffer at the start of epoch if necessary
        """
        if self.cfg.use_persistent_sample_buffer:
            self.persistent_sample_buffer =  self.replenish_sample_buffer(
                num_trajectories=self.cfg.persistent_sample_buffer_trajectories,
                proposal_type=self.proposal_type,
                T=self.T,
            )
            if self.cfg.verbose:
                buffer_size = self.persistent_sample_buffer["xs"].shape[0]
                print(f'Replenished sample buffer with {buffer_size} trajectories')
        else:
            self.persistent_sample_buffer = None

    def training_step(self, batch, batch_idx):
        start_bytes = torch.cuda.memory_allocated()
        if self.cfg.use_persistent_sample_buffer:
            pinn_loss = self.pinn_loss_wrapper(
                num_trajectories = self.cfg.train_trajectories,
                sample_buffer=self.persistent_sample_buffer,
            )
        else:
            pinn_loss = self.pinn_loss_wrapper(
                num_trajectories=self.cfg.train_trajectories,
                proposal_type=self.proposal_type,
                T=self.T,
            )
        end_bytes = torch.cuda.memory_allocated()
        byte_diff_gib = (end_bytes - start_bytes) / BYTES_PER_GIB
        self.train_memory_usages.append(byte_diff_gib)
        self.train_losses.append(pinn_loss.item())
        return pinn_loss
    
    def on_train_epoch_end(self):
        # Report average training loss
        self.log(
            "Train Loss", np.mean(self.train_losses), on_epoch=True, prog_bar=True, logger=True
        )
        self.log("Memory Demand (GiB)", max(self.train_memory_usages), on_epoch=True, prog_bar=True)
        self.log("T", self.T, on_epoch=True, prog_bar=True, logger=True)

        # Reset metrics for next epoch
        self.train_losses = []
        self.train_memory_usages = []

        # Update PINN-annealing parameter
        self.annealing_scheduler.step()

    def validation_step(self, batch, batch_idx):
        pass
        # pinn_loss = self.pinn_loss_wrapper(
        #     batch_size=self.cfg.val_batch_size,
        #     num_trajectories=self.cfg.val_trajectories,
        #     proposal_type=self.proposal_type,
        #     T=1.0,
        # )
        # self.val_losses.append(pinn_loss)
        # return pinn_loss

    def on_validation_epoch_end(self):
        # Report average validation loss
        # avg_val_loss = torch.mean(torch.stack(self.val_losses))
        # self.log(
        #     "val_loss",
        #     avg_val_loss,
        #     on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     sync_dist=True,
        # )
        # self.val_losses = []

        # Report additional metrics
        ode_proposal = self.build_proposal("ode")
        ode_ts = torch.linspace(0, 1, 250).to(get_module_device(self)).view(1, -1, 1).expand(self.cfg.val_trajectories, -1, 1)
        metrics = ode_proposal.get_metrics(
            ts=ode_ts,
            label="val",
        )
        for key, value in metrics.items():
            self.log(key, value, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        def lr_lambda(epoch):
            if epoch < self.cfg.lr_burn_in_epochs:
                return 1.0  # No change to LR
            # Adjust LR as StepLR would
            step_count = (epoch - self.cfg.lr_burn_in_epochs) // self.cfg.step_size
            return self.cfg.gamma ** step_count

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def val_dataloader(self):
        return DummyDataloader(self.cfg.val_steps_per_epoch)

    def train_dataloader(self):
        return DummyDataloader(self.cfg.train_steps_per_epoch)
