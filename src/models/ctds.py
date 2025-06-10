import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, List
from multiprocessing import Pool

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import LightningModule
from torch import Tensor
from tqdm import tqdm
import wandb
import numpy as np

from src.datasets import DummyDataloader
from src.models.utils import AnnealingScheduler
from src.simulation.dynamics import (
    AuxiliaryProcess,
    ODEProcess
)
from src.simulation.langevin import (
    UnderdampedCTDS,
    MTAnnealedHamiltonian,
    build_mt_annealed_hamiltonian,
)
from src.simulation.vector_field import ConditionalMLPVectorField, ZeroConditionalVectorField, ConditionalVectorField
from src.systems.base import Sampleable, Uniform, BumpInterval
from src.systems.density_path import ConstantDensityPath
from src.systems.mt import (
    MultiTempVectorFieldWrapper,
    FixedTempVectorFieldWrapper,
    MLPMTContinuumFreeEnergy,
    MTContinuum,
    MTContinuumFreeEnergy,
    ReparameterizedConditionalVectorField,
    MTSampleable,
    BetaConverter,
    ReparameterizedMTContinuum,
    WrapperReparameterizedMTContinuum,
    ZeroMTContinuumFreeEnergy,
    build_biasing_density,
    build_converter,
    build_mt_continuum,
)
from src.models.nets import PINNProposal, ODEProposal
from src.utils.metrics import (
    MAX_LOG_WEIGHT,
)
from src.utils.misc import get_module_device

#################################
# Jarzynski Auxiliary Processes #
#################################
    
class CTDSJarzynski(AuxiliaryProcess):
    """
    Auxiliary process for the Jarzynski weights of any mixed process for generic augmenting Langevin dynamics
    """
    def __init__(self, x_dim: int, x_control: ConditionalVectorField, xz_continuum: ReparameterizedMTContinuum, converter: BetaConverter, divergence_mode: str):
        super().__init__()
        self.x_dim = x_dim
        self.x_control = x_control
        self.xz_continuum = xz_continuum
        self.divergence_mode = divergence_mode

    def initial_value(self, num_samples: int) -> Tensor:
        return torch.zeros(num_samples, 1)
    
    def integrate_step(
            self, at: Tensor, q: Tensor, t: Tensor, dt: Tensor
    ) -> Tensor:
        """
        Args:
        - at: (bs, 1)
        - qt: (bs, 2 * x_dim + 2) or (bs, x_dim + 1)
        - t: (bs, 1)
        - dt: (bs, 1)
        """
        xz = q[:, : self.x_dim + 1]
        x = xz[:, : self.x_dim]
        z = xz[:, self.x_dim : self.x_dim + 1]

        # Compute divergence wrt x
        div = self.x_control.divergence(x, z, t, mode=self.divergence_mode)  # (batch_size, 1)

        # Note: we exclude the last coordinate from the log density because the control is zero for z
        dxz_log_pt = self.xz_continuum.dxz_log_density(xz, t)[:,:-1] # (batch_size, x_dim)
        control = self.x_control(x, z, t)  # (batch_size, x_dim)
        dot = (dxz_log_pt * control).sum(dim=-1, keepdim=True) # (batch_size, 1)

        dt_log_pt = self.xz_continuum.dt_log_density(xz, t) # (batch_size, 1)

        return at + (div + dot + dt_log_pt) * dt


#########################################
# Continuously Tempered Model Proposals #
#########################################


class MTPINNProposal(nn.Module, ABC):
    @abstractmethod
    def sample(self, ts: Tensor, **kwargs) -> Dict:
        raise NotImplementedError

    
class MTODEProposal(MTPINNProposal):
    def __init__(
        self,
        control: ConditionalVectorField,
        source_sampleable: MTSampleable,
        record_every: int,
        divergence_mode: str
    ):
        super().__init__()
        self.dynamics = ODEProcess(
            source=source_sampleable,
            vector_field=control
        )
        self.record_every = record_every
        self.x_dim = source_sampleable.dim - 1 # last dimension is beta

    def sample(self, ts, **kwargs):
        batch_size = ts.shape[0]
        trajectory, _, ts = self.dynamics.sample_with_trajectory(
            ts=ts, num_samples=batch_size, record_every=self.record_every, **kwargs
        )
        xbs = trajectory[:, :, : self.x_dim + 1].detach().reshape(-1, self.x_dim + 1)
        xs = xbs[:, :-1]
        bs = xbs[:, -1:]
        ts = ts.detach().reshape(-1, 1)

        return {
            "xs": xs,  # (batch_size * num_integration_steps + 1, x_dim + 1)
            "bs": bs,  # (batch_size * num_integration_steps + 1, 1)
            "ts": ts,  # (batch_size * num_integration_steps + 1, 1)
            "log_weights": torch.zeros_like(ts).detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
            "weights": torch.ones_like(ts).detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
        }


class UniformMTPINNProposal(MTPINNProposal):
    def __init__(
        self, x_dim: int, scale: float, min_z: float = 0.2, max_z: float = 1.0
    ):
        """
        Args:
        - x_dim: int, dimension of the data
        - scale: float, scale of the uniform distribution
        - min_z: float, minimum beta value
        - max_z: float, maximum beta value
        """
        super().__init__()
        self.x_dim = x_dim
        self.scale = scale
        self.min_z = min_z
        self.max_z = max_z


    def sample(self, ts: Tensor, **kwargs) -> Dict:
        """
        Sample x-coordinates uniformly from [-scale, scale] ^ data_dim, and intended mostly as a debugging tool for low-dimensional systems, but is used in e.g., https://arxiv.org/abs/2407.07873
        Args
        - ts: (batch_size, num_integration_steps + 1, 1)
        Returns Dict with keys:
        - xzs: (batch_size * num_integration_steps + 1, x_dim+1)
        - ts: (batch_size *, num_integration_steps + 1, 1)
        - weights: (batch_size * num_integration_steps + 1, 1)
        - log_weights: (batch_size * num_integration_steps + 1, 1)
        """
        batch_size, num_ts, _ = ts.shape
        xts = self.scale * (2 * torch.rand(batch_size, num_ts, self.x_dim) - 1).to(
            ts
        )  # (bs, num_integration_steps, data_dim)
        bts = (
            torch.rand(batch_size, num_ts, 1) * (self.max_beta - self.min_beta)
            + self.min_beta
        )
        bts = bts.to(ts)
        xbs = torch.cat([xts, bts], dim=-1)
        return {
            "xbs": xbs.detach().reshape(-1, self.x_dim + 1),  # (batch_size * num_integration_steps + 1, x_dim + 1)
            "ts": ts.detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
            "weights": torch.ones_like(bts).detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
            "log_weights": torch.zeros_like(bts).detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
        }

class ForwardProcessProposal(MTPINNProposal):
    def sample(self, ts: Tensor, xz_pxz0: Optional[Tensor] = None, **kwargs) -> Dict:
        """
        Args: 
        - ts: (batch_size, num_integration_steps + 1, 1)
        - xz_pxz_0: (batch_size, x_dim + 1)
        Returns Dict with keys:
        - xs: (batch_size * num_integration_steps + 1, x_dim+1)
        - zs: (batch_size * num_integration_steps + 1, 1)
        - bs: (batch_size * num_integration_steps + 1, 1)
        - ts: (batch_size *, num_integration_steps + 1, 1)
        - weights: (batch_size * num_integration_steps + 1, 1)
        - log_weights: (batch_size * num_integration_steps + 1, 1)
        """
        trajectory, aux_trajectory, ts = self.dynamics.sample_with_trajectory(
            ts=ts, num_samples=ts.shape[0], record_every=self.record_every, x0=xz_pxz0, use_tqdm=True, **kwargs
        )
        xs = trajectory[:, :, : self.x_dim]  # (batch_size, num_integration_steps + 1, x_dim)
        zs = trajectory[:, :, self.x_dim : self.x_dim + 1]  # (batch_size, num_integration_steps + 1, 1)
        bs = self.converter.xi_to_beta(zs)
        if self.use_jarzynski:
            log_weights = aux_trajectory["jarzynski"]  # (batch_size, num_integration_steps + 1, 1)
            weights = torch.clamp(log_weights, max=MAX_LOG_WEIGHT)  # (batch_size, num_integration_steps + 1, 1)
            weights = torch.exp(weights)  # (batch_size, num_integration_steps + 1, 1)
            weights = weights / torch.sum(weights, dim=0, keepdim=True)  # (bs, num_integration_steps + 1, 1)
        else:
            weights = torch.ones_like(ts)
            log_weights = torch.zeros_like(ts)
        return {
            "xs": xs.detach().reshape(-1, self.x_dim),  # (batch_size * num_integration_steps + 1, x_dim + 1)
            "zs": zs.detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
            "bs": bs.detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
            "ts": ts.detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
            "weights": weights.detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
            "log_weights": log_weights.detach().reshape(-1, 1),  # (batch_size * num_integration_steps + 1, 1)
        }

class OverdampedCTDSProposal(ForwardProcessProposal):
    """
    Overdamped x dynamics, underdamped z dynamics
    """
    pass

class UnderdampedCTDSProposal(ForwardProcessProposal):
    """
    Underdamped x dynamics, underdamped z dynamics
    """
    def __init__(
        self,
        x_control: ConditionalVectorField,
        x_scale: float,
        z_scale: float,
        x_damping: float,
        z_damping: float,
        source: Sampleable,
        hamiltonian: MTAnnealedHamiltonian,
        enforce_z_bounds: bool,
        z_lower_bound: float,
        z_upper_bound: float,
        divergence_mode: str,
        record_every: int = 1,
        use_jarzynski: bool = False
    ):
        super().__init__()
        self.xz_continuum = hamiltonian.xz_continuum
        self.converter = self.xz_continuum.converter
        self.z_biasing_density = self.xz_continuum.z_biasing_density
        aux_processes = {}
        if use_jarzynski:
            aux_dynamics = CTDSJarzynski(
                x_dim=self.xz_continuum.x_dim, x_control=x_control, xz_continuum=self.xz_continuum, converter=self.converter, divergence_mode=divergence_mode
            )
            aux_processes = {"jarzynski": aux_dynamics}
        self.dynamics = UnderdampedCTDS(
            x_control=x_control,
            x_scale = x_scale,
            z_scale = z_scale,
            x_damping=x_damping,
            z_damping=z_damping,
            source=source,
            hamiltonian=hamiltonian,
            converter=self.converter,
            enforce_z_bounds=enforce_z_bounds,
            z_lower_bound=z_lower_bound,
            z_upper_bound=z_upper_bound,
            auxiliary_processes=aux_processes
        )
        self.record_every = record_every
        self.x_dim = self.xz_continuum.x_dim
        self.use_jarzynski = use_jarzynski

    def sample(self, ts: Tensor, xz_pxz0: Optional[Tensor] = None, **kwargs) -> Dict:
        return super().sample(ts=ts, xz_pxz0=xz_pxz0, **kwargs)


class CTDSModule(LightningModule):
    """
    General fromework for PINN-based training in the multi-temperature setting.
    """

    def __init__(
        self,
        cfg: DictConfig,
        x_control: Optional[ZeroConditionalVectorField] = None,
        mt_continuum: Optional[MTContinuum] = None,
        free_energy: Optional[MTContinuumFreeEnergy] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["x_control", "mt_continuum", "free_energy"])
        self.cfg = cfg

        # Continuum
        if mt_continuum is None:
            self.mt_continuum = build_mt_continuum(cfg)
        else:
            self.mt_continuum = mt_continuum
        self.x_dim = self.mt_continuum.x_dim

        # Control
        if x_control is not None:
            self.x_control = x_control
        elif cfg.x_control == "mlp":
            if cfg.use_fourier:
                self.x_control = ConditionalMLPVectorField(
                    data_dim=self.x_dim,
                    hidden_dims=cfg.x_control_hiddens,
                    conditioning_dim=1, 
                    use_fourier=True,
                    x_fourier_dim=cfg.x_fourier_dim,
                    x_fourier_sigma=cfg.x_fourier_sigma,
                    c_fourier_dim=cfg.b_fourier_dim,
                    c_fourier_sigma=cfg.b_fourier_sigma,
                    t_fourier_dim=cfg.t_fourier_dim,
                    t_fourier_sigma=cfg.t_fourier_sigma,
                )
            else:
                self.x_control = ConditionalMLPVectorField(
                    data_dim=self.x_dim,
                    hidden_dims=cfg.x_control_hiddens,
                    conditioning_dim=1, 
                    use_fourier=False,
                )
        elif cfg.x_control == "zero":
            self.x_control = ZeroConditionalVectorField()

        if free_energy is not None:
            self.free_energy = free_energy
        elif cfg.free_energy == "mlp":
            if cfg.use_fourier:
                self.free_energy = MLPMTContinuumFreeEnergy(
                    cfg.free_energy_hiddens,
                    use_fourier=True,
                    b_fourier_dim=cfg.b_fourier_dim,
                    b_fourier_sigma=cfg.b_fourier_sigma,
                    t_fourier_dim=cfg.t_fourier_dim,
                    t_fourier_sigma=cfg.t_fourier_sigma,
                )
            else:
                self.free_energy = MLPMTContinuumFreeEnergy(
                    cfg.free_energy_hiddens,
                    use_fourier=False,
                )
        elif cfg.free_energy == "zero":
            self.free_energy = ZeroMTContinuumFreeEnergy()

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

        # Maintain stateful proposal (statefulenuss is necessary to implement e.g., biasing density)
        self.proposal = self.build_mt_proposal(cfg.proposal)

    def move_to_device(self, device: torch.device, verbose=False):
        """
        For use switching devices when working in e.g., a Jupyter notebook.
        """
        for name, child in self.named_children():
            child.to(device)
            if verbose:
                print(f"Moving {name} to {self.device}")
        return self

    @property
    def T(self) -> float:
        return self.annealing_scheduler.T

    def setup(self, stage):
        # Move all submodules to device
        for child in self.children():
            child.to(self.device)
        # Initialize losses
        self.train_losses = []
        self.val_losses = []

    def pinn_loss(self, x:Tensor, b: Tensor, t: Tensor, weights: Tensor):
        """
        Compute the MT PINN loss up to time T given fixed  samples
        Args:
            xb: (batch_size, data_dim + 1)
            t: (batch_size, 1)
            weights: (batch_size, 1)
        Returns:
            pinn_loss: (1)
        """

        xb = torch.cat([x, b], dim=-1)  # (batch_size, data_dim + 1)

        # \partial_t F(b,t)
        dt_Fbt = self.free_energy.dt_free_energy_at(b, t)  # (batch_size, 1)

        # control(x,t) \dot \nabla_x ln p_t(xt)
        x_control = self.x_control(x, b, t)  # (batch_size, data_dim)
        x_score = self.mt_continuum.dxb_log_density(xb, t)[:, :-1]
        dot = (x_control * x_score).sum(-1, keepdims=True)  # (batch_size, 1)

        # \nabla \cdot control(x,t)
        div = self.x_control.divergence(
            x, b, t, mode=self.cfg.divergence_mode
        )  # (batch_size, 1)

        # \partial_t \ln p(x,t)
        dt_ln_pt = self.mt_continuum.dt_log_density(xb, t)  # (batch_size, 1)

        # Compute raw loss
        raw_loss = (dt_Fbt + div + dot + dt_ln_pt) ** 2
        return torch.mean(weights * raw_loss)

    def get_integration_ts(self, num_trajectories: int, T: float) -> Tensor:
        """
        Construct integration steps by which to discretize MT PINN integral
        Args:
        - num_samples: int, the number of samples
        - T: float, upper bound of integration
        Returns:
        - ts: (num_samples, nts, 1)
        """
        num_integration_steps = math.ceil(T / self.integration_avg_dt)
        integration_ts = T * torch.sort(torch.rand(num_integration_steps - 1)).values
        integration_ts = torch.cat([torch.zeros(1), integration_ts, torch.ones(1) * T])
        return (
            integration_ts.view(1,-1,1)
            .expand(num_trajectories, -1, 1)
            .to(get_module_device(self))
        )
    
    def build_underdamped_ctds_proposal(
        self,
    ) -> UnderdampedCTDSProposal:
        z_biasing_density = build_biasing_density(self.cfg)
        self.converter = build_converter(self.cfg)
        x_control = ReparameterizedConditionalVectorField(
            vector_field=self.x_control,
            converter=self.converter,
        )
        z_confining_distribution = BumpInterval(
            low=self.cfg.confining_lower_bound, high=self.cfg.confining_upper_bound, sharpness=self.cfg.confining_sharpness
        )
        z_confining_density_path = ConstantDensityPath(
            z_confining_distribution, z_confining_distribution
        )
        xz_continuum = WrapperReparameterizedMTContinuum(
            mt_continuum=self.mt_continuum,
            converter=self.converter,
            z_biasing_density=z_biasing_density,
            bt_free_energy=self.free_energy,
            z_confining_density_path=z_confining_density_path,
            z_source=z_confining_density_path.start_sampleable,
        )
        
        # Build hamiltonian
        hamiltonian = build_mt_annealed_hamiltonian(
            xz_continuum=xz_continuum,
            hamiltonian_type=self.cfg.hamiltonian_type,
            x_mass=self.cfg.x_mass,
            z_mass=self.cfg.z_mass,
        )

        # Build source distribution
        source = hamiltonian.get_start_sampleable()

        # Finally, construct and return process
        return UnderdampedCTDSProposal(
            x_control=x_control,
            x_scale=self.cfg.x_scale,
            z_scale=self.cfg.z_scale,
            x_damping=self.cfg.x_damping,
            z_damping=self.cfg.z_damping,
            source=source,
            hamiltonian=hamiltonian,
            enforce_z_bounds=False,
            z_lower_bound=self.cfg.z_lower_bound,
            z_upper_bound=self.cfg.z_upper_bound,
            divergence_mode=self.cfg.divergence_mode,
            record_every=self.cfg.record_every,
            use_jarzynski=self.cfg.use_jarzynski,
        )


    def build_mt_proposal(
        self, proposal_type: str, **kwargs
    ) -> MTPINNProposal:
        """
        Build MTProposal
        """
        if proposal_type == "overdamped_ctds":
            raise NotImplementedError
        elif proposal_type == "underdamped_ctds":
            return self.build_underdamped_ctds_proposal()
        elif proposal_type == "ode":
            source_sampleable = MTSampleable(
                x_sampleable=self.mt_continuum.reference_start_sampleable,
                beta_sampleable=Uniform(low=self.cfg.beta_min, high=self.cfg.beta_max, dim=1),
            )
            return MTODEProposal(
                control=MultiTempVectorFieldWrapper(self.x_control),
                source_sampleable=source_sampleable,
                record_every=self.cfg.record_every,
                divergence_mode=self.cfg.divergence_mode,
            )
        else:
            raise ValueError(f"Unknown proposal type: {proposal_type}")

    def build_proposal(self, proposal_type: str, **kwargs) -> PINNProposal:
        """
        Build non-temperature-mixed PINN proposal at the reference temperature
        """
        if proposal_type == "ode":
            return ODEProposal(
                control=FixedTempVectorFieldWrapper(self.x_control),
                source_sampleable=self.mt_continuum.reference_start_sampleable,
                record_every=self.cfg.record_every,
                divergence_mode=self.cfg.divergence_mode,
                target_sampleable=self.mt_continuum.reference_end_sampleable
            )
        else:
            raise ValueError(f"Unknown proposal type: {proposal_type}")
        
    def replenish_sample_buffer(self, num_trajectories: int, integration_ts: Optional[Tensor] = None, T: float = 1.0) -> Dict:
        if integration_ts is None:
            integration_ts = self.get_integration_ts(num_trajectories, T)
        else:
            integration_ts = integration_ts.view(1, -1, 1).expand(num_trajectories, -1, 1).to(get_module_device(self))

        return self.proposal.sample(ts=integration_ts)

    def pinn_loss_wrapper(
        self,
        batch_size: int,
        sample_buffer: Optional[Dict] = None,
        num_trajectories: Optional[int] = None,
        integration_ts: Optional[Tensor] = None,
        T: float = 1.0,
    ):
        """
        Sample trajectories from proposal and call compute off-policy PINN loss
        Args:
        - num_samples: int, the number of samples to take
        - integration_ts: Optional[Tensor], the integration timesteps (nts, 1)
        Returns:
        - pinns_loss: (1)
        """
        # Construct integration steps by which to discretize PINN integral if necessary and expand to add batch dimension given by num_samples
        if sample_buffer is None:
            assert num_trajectories is not None 
            sample_buffer = self.replenish_sample_buffer(
                num_trajectories=num_trajectories,
                integration_ts=integration_ts,
                T=T
            )

        # Subsample to batch size
        buffer_size = sample_buffer["xs"].shape[0]
        subsample_idxs = torch.randint(buffer_size, (batch_size,))
        xs = sample_buffer["xs"][subsample_idxs]  # (bs, data_dim)
        bs = sample_buffer["bs"][subsample_idxs]  # (bs, 1)
        ts = sample_buffer["ts"][subsample_idxs]  # (bs, 1)
        weights = sample_buffer["weights"][subsample_idxs]  # (bs, 1)

        return self.pinn_loss(xs, bs, ts, weights)

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
                T=self.T,
            )
            if self.cfg.verbose:
                buffer_size = self.persistent_sample_buffer["xs"].shape[0]
                print(f'Replenished sample buffer with {buffer_size} trajectories')
        else:
            self.persistent_sample_buffer = None

    def training_step(self, batch, batch_idx):
        if self.cfg.use_persistent_sample_buffer:
            pinn_loss = self.pinn_loss_wrapper(
                batch_size = self.cfg.train_batch_size,
                sample_buffer=self.persistent_sample_buffer,
            )
        else:
            pinn_loss = self.pinn_loss_wrapper(
                batch_size=self.cfg.train_batch_size,
                num_trajectories=self.cfg.train_trajectories,
                T=self.T,
            )
        self.train_losses.append(pinn_loss.item())
        return pinn_loss

    def validation_step(self, batch, batch_idx):
        # pinn_loss = self.pinn_loss_wrapper(
        #     batch_size=self.cfg.val_batch_size,
        #     num_trajectories=self.cfg.val_trajectories,
        #     T=1.0,
        # )
        # self.val_losses.append(pinn_loss)
        # return pinn_loss
        pass

    def on_train_epoch_end(self):
        # Report loss-related metrics
        avg_train_loss = np.mean(self.train_losses)
        self.log(
            "train_loss", avg_train_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.train_losses = []

        # Log PINN-annealing parameter and update
        self.log("T", self.T, on_epoch=True, prog_bar=True, logger=True)
        self.annealing_scheduler.step()

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

        proposal = self.build_proposal("ode")
        ts = self.get_integration_ts(num_trajectories=self.cfg.val_trajectories, T=1.0)
        proposal_metrics = proposal.get_metrics(ts, label="val")
        for key, value in proposal_metrics.items():
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
