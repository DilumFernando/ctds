from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base import module_device
from .config import Config
from .density_paths import DensityPath, build_density_path
from .dynamics import AuxiliaryProcess, DivegenceAuxiliaryProcess, ODEProcess
from .langevin import AnnealedOverdampedLangevin
from .metrics import MAX_LOG_WEIGHT, ess, normalize_log_weights, w2_from_samples
from .nn import FeedForward, GaussianFourierEncoder
from .vector_fields import MLPVectorField, VectorField

BYTES_PER_GIB = 1024 * 1024 * 1024


def cuda_profile(fn):
    @wraps(fn)
    def wrapper(*args, profile: bool = False, **kwargs):
        if profile and torch.cuda.is_available():
            start_bytes = torch.cuda.memory_allocated()
            result = fn(*args, **kwargs)
            end_bytes = torch.cuda.memory_allocated()
            gib = (end_bytes - start_bytes) / BYTES_PER_GIB
            print(f"Call to {fn.__name__} used {gib:.3f} GiB of memory")
            return result
        return fn(*args, **kwargs)

    return wrapper


class DtFreeEnergy(nn.Module, ABC):
    @abstractmethod
    def forward(self, time: Tensor) -> Tensor:
        raise NotImplementedError


class MLPDtFreeEnergy(DtFreeEnergy):
    def __init__(
        self,
        hiddens: list[int],
        use_fourier: bool,
        t_fourier_dim: int,
        t_fourier_sigma: float,
    ):
        super().__init__()
        if use_fourier:
            self.t_encoder = GaussianFourierEncoder(1, t_fourier_dim, t_fourier_sigma)
            self.net = nn.Sequential(self.t_encoder, FeedForward([t_fourier_dim] + hiddens + [1]))
        else:
            self.net = FeedForward([1] + hiddens + [1])

    @cuda_profile
    def forward(self, time: Tensor) -> Tensor:
        return self.net(time)


class OverdampedJarzynski(AuxiliaryProcess):
    def __init__(self, control: VectorField, density_path: DensityPath, divergence_mode: str):
        super().__init__()
        self.control = control
        self.density_path = density_path
        self.divergence_mode = divergence_mode

    def initial_value(self, num_samples: int) -> Tensor:
        return torch.zeros(num_samples, 1)

    def integrate_step(self, at: Tensor, xt: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        div = self.control.divergence(xt, t, self.divergence_mode)
        dx_log_pt = self.density_path.dx_log_density(xt, t)
        dot = (dx_log_pt * self.control(xt, t)).sum(dim=-1, keepdim=True)
        dt_log_pt = self.density_path.dt_log_density(xt, t)
        return at + (div + dot + dt_log_pt) * dt


class PINNProposal(nn.Module, ABC):
    @abstractmethod
    def sample(self, ts: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class ODEProposal(PINNProposal):
    def __init__(
        self,
        control: VectorField,
        source_sampleable,
        target_sampleable,
        record_every: int,
        divergence_mode: str,
    ):
        super().__init__()
        self.auxiliary_processes = {"cov": DivegenceAuxiliaryProcess(control, divergence_mode)}
        self.source_sampleable = source_sampleable
        self.target_sampleable = target_sampleable
        self.dynamics = ODEProcess(source_sampleable, control, self.auxiliary_processes)
        self.record_every = record_every
        self.x_dim = source_sampleable.dim

    def sample(self, ts: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        trajectory, aux_trajectory, ts = self.dynamics.sample_with_trajectory(
            ts=ts,
            num_samples=ts.shape[0],
            record_every=self.record_every,
            **kwargs,
        )
        x0 = trajectory[:, 0, : self.x_dim]
        init_log_weights = self.source_sampleable.log_density(x0).view(-1, 1, 1)
        log_weights = init_log_weights + aux_trajectory["cov"]
        return trajectory, ts, log_weights

    def get_metrics(self, ts: Tensor, label: str) -> Dict[str, float]:
        trajectory, ts, log_weights = self.sample(ts)
        model_samples = trajectory[:, -1, : self.x_dim]
        log_model_weights = log_weights[:, -1, :]
        log_target_weights = self.target_sampleable.log_density(model_samples)
        metrics = {f"{label}_elbo": torch.mean(log_target_weights - log_model_weights).item()}

        target_samples = self.target_sampleable.sample(model_samples.shape[0]).to(model_samples)
        metrics[f"{label}_w2"] = w2_from_samples(model_samples, target_samples).item()
        model_sample_weights = normalize_log_weights(log_model_weights).to(model_samples)
        target_sample_weights = torch.ones(target_samples.shape[0], device=target_samples.device) / target_samples.shape[0]
        metrics[f"{label}_weighted_w2"] = w2_from_samples(
            model_samples,
            target_samples,
            p_weights=model_sample_weights,
            q_weights=target_sample_weights,
        ).item()

        reverse_ts = ts.flip(1)
        x0, aux_dict = self.dynamics.sample(ts=reverse_ts, x0=target_samples)
        log_model_weights = self.source_sampleable.log_density(x0) - aux_dict["cov"]
        log_target_weights = self.target_sampleable.log_density(target_samples)
        metrics[f"{label}_r_ess"] = ess(log_model_weights - log_target_weights).item()
        metrics[f"{label}_eubo"] = torch.mean(log_target_weights - log_model_weights).item()
        return metrics


class OverdampedLangevinProposal(PINNProposal):
    def __init__(
        self,
        control: VectorField,
        damping: float,
        density_path: DensityPath,
        divergence_mode: str,
        record_every: int,
        use_jarzynski: bool,
    ):
        super().__init__()
        jarzynski = OverdampedJarzynski(control, density_path, divergence_mode)
        aux_processes = {"jarzynski": jarzynski} if use_jarzynski else {}
        self.dynamics = AnnealedOverdampedLangevin(control, damping, density_path, aux_processes)
        self.use_jarzynski = use_jarzynski
        self.record_every = record_every
        self.x_dim = density_path.dim

    def sample(self, ts: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        trajectory, aux_trajectory, ts = self.dynamics.sample_with_trajectory(
            ts=ts,
            num_samples=ts.shape[0],
            record_every=self.record_every,
            use_tqdm=kwargs.pop("use_tqdm", True),
            **kwargs,
        )
        trajectory = trajectory[:, :, : self.x_dim]
        if self.use_jarzynski:
            log_weights = aux_trajectory["jarzynski"]
        else:
            log_weights = torch.zeros_like(ts)
        return trajectory, ts, log_weights


class ManualAnnealingScheduler:
    def __init__(self, T_schedule: list[float], epochs_per_T: list[float]):
        self.T_schedule = T_schedule
        self.epochs_per_T = epochs_per_T
        self.current_idx = 0
        self.epochs_since_update = 0
        self._T = T_schedule[0]

    @property
    def T(self) -> float:
        return self._T

    def step(self) -> None:
        if self.current_idx >= len(self.T_schedule):
            return
        self.epochs_since_update += 1
        if self.epochs_since_update >= self.epochs_per_T[self.current_idx]:
            self.current_idx += 1
            if self.current_idx < len(self.T_schedule):
                self._T = self.T_schedule[self.current_idx]
                self.epochs_since_update = 0


class NETSModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.x_dim = int(cfg.x_dim)
        self.density_path = build_density_path(cfg)
        self.control = MLPVectorField(
            data_dim=self.x_dim,
            hidden_dims=list(cfg.control_hiddens),
            use_fourier=bool(cfg.use_fourier),
            x_fourier_dim=int(cfg.x_fourier_dim),
            x_fourier_sigma=float(cfg.x_fourier_sigma),
            t_fourier_dim=int(cfg.t_fourier_dim),
            t_fourier_sigma=float(cfg.t_fourier_sigma),
        )
        self.dt_free_energy = MLPDtFreeEnergy(
            hiddens=list(cfg.free_energy_hiddens),
            use_fourier=bool(cfg.use_fourier),
            t_fourier_dim=int(cfg.t_fourier_dim),
            t_fourier_sigma=float(cfg.t_fourier_sigma),
        )
        self.annealing_scheduler = ManualAnnealingScheduler(
            list(cfg.T_schedule),
            list(cfg.epochs_per_T),
        )
        self.integration_avg_dt = float(cfg.avg_dt)

    @property
    def T(self) -> float:
        return self.annealing_scheduler.T

    def get_integration_ts(self, T: float) -> Tensor:
        num_integration_steps = math.ceil(T / self.integration_avg_dt)
        integration_ts = T * torch.sort(torch.rand(num_integration_steps - 1, device=module_device(self))).values
        integration_ts = torch.cat(
            [
                torch.zeros(1, device=integration_ts.device),
                integration_ts,
                torch.ones(1, device=integration_ts.device) * T,
            ]
        )
        return integration_ts.unsqueeze(-1)

    def build_proposal(self, proposal_type: str) -> PINNProposal:
        if proposal_type == "overdamped_langevin":
            return OverdampedLangevinProposal(
                control=self.control,
                damping=float(self.cfg.damping),
                density_path=self.density_path,
                record_every=int(self.cfg.record_every),
                use_jarzynski=bool(self.cfg.use_jarzynski),
                divergence_mode=str(self.cfg.divergence_mode),
            )
        if proposal_type == "ode":
            return ODEProposal(
                control=self.control,
                source_sampleable=self.density_path.start_sampleable,
                target_sampleable=self.density_path.end_sampleable,
                record_every=int(self.cfg.record_every),
                divergence_mode=str(self.cfg.divergence_mode),
            )
        raise NotImplementedError("Standalone repo only includes overdamped_langevin and ode proposals.")

    def extract_weights(self, log_weights: Tensor) -> Tensor:
        log_weights = torch.clamp(log_weights, max=MAX_LOG_WEIGHT)
        weights = torch.exp(log_weights)
        return weights / torch.mean(weights, dim=0, keepdim=True)

    def replenish_sample_buffer(self, num_trajectories: int, proposal_type: str, T: float) -> Dict[str, Tensor]:
        integration_ts = self.get_integration_ts(T).unsqueeze(0).expand(num_trajectories, -1, 1)
        proposal = self.build_proposal(proposal_type).to(module_device(self))
        xs, ts, log_weights = proposal.sample(ts=integration_ts)
        weights = self.extract_weights(log_weights)
        return {
            "xs": xs.detach(),
            "ts": ts.detach(),
            "log_weights": log_weights.detach(),
            "weights": weights.detach(),
        }

    def pinn_loss(self, x: Tensor, t: Tensor, weights: Tensor) -> Tensor:
        dt_Ft = self.dt_free_energy(t, profile=bool(self.cfg.memory_profile))
        control = self.control(x, t)
        score = self.density_path.dx_log_density(x, t, profile=bool(self.cfg.memory_profile))
        div = self.control.divergence(
            x,
            t,
            str(self.cfg.divergence_mode),
            profile=bool(self.cfg.memory_profile),
        )
        dt_ln_pt = self.density_path.dt_log_density(x, t, profile=bool(self.cfg.memory_profile))
        raw_loss = (dt_Ft + div + (control * score).sum(-1, keepdims=True) + dt_ln_pt) ** 2
        return torch.mean(weights * raw_loss)

    def compute_train_loss(self, sample_buffer: Optional[Dict[str, Tensor]]) -> Tensor:
        if sample_buffer is not None:
            subsample_idxs = torch.randperm(sample_buffer["xs"].shape[0], device=sample_buffer["xs"].device)[
                : int(self.cfg.train_trajectories)
            ]
            xs = sample_buffer["xs"][subsample_idxs]
            ts = sample_buffer["ts"][subsample_idxs]
            weights = sample_buffer["weights"][subsample_idxs]
        else:
            sample_buffer = self.replenish_sample_buffer(
                num_trajectories=int(self.cfg.train_trajectories),
                proposal_type="overdamped_langevin",
                T=self.T,
            )
            xs = sample_buffer["xs"]
            ts = sample_buffer["ts"]
            weights = sample_buffer["weights"]

        return self.pinn_loss(
            xs.reshape(-1, self.x_dim).detach().clone(),
            ts.reshape(-1, 1).detach().clone(),
            weights.reshape(-1, 1).detach().clone(),
        )

    def compute_validation_loss(self) -> Tensor:
        sample_buffer = self.replenish_sample_buffer(
            num_trajectories=int(self.cfg.val_trajectories),
            proposal_type="overdamped_langevin",
            T=1.0,
        )
        return self.pinn_loss(
            sample_buffer["xs"].reshape(-1, self.x_dim).detach().clone(),
            sample_buffer["ts"].reshape(-1, 1).detach().clone(),
            sample_buffer["weights"].reshape(-1, 1).detach().clone(),
        )

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        device = module_device(self)
        val_loss = self.compute_validation_loss().item()
        proposal = self.build_proposal("ode").to(device)
        ode_ts = torch.linspace(0, 1, 250, device=device).view(1, -1, 1)
        ode_ts = ode_ts.expand(int(self.cfg.val_trajectories), -1, 1)
        metrics = proposal.get_metrics(ts=ode_ts, label="val")
        metrics["val_loss"] = float(val_loss)
        return metrics

    def memory_delta_gib(self, start_bytes: int, end_bytes: int) -> float:
        return (end_bytes - start_bytes) / BYTES_PER_GIB
