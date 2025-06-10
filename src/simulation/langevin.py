import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.func import jacrev, vmap
from tqdm import tqdm
from omegaconf import DictConfig

from src.systems.base import ProductSampleable, Sampleable, Density, WrapperSampleable
from src.systems.density_path import DensityPath
from src.systems.distributions import Gaussian
from src.simulation.vector_field import VectorField, ConditionalVectorField
from src.simulation.dynamics import ForwardDiffusionProcess, AuxiliaryProcess
from src.systems.mt import (
    BetaConverter,
    ReparameterizedMTContinuum
)

#####################
# Langevin Dynamics #
#####################

class OverdampedLangevin(ForwardDiffusionProcess):
    def __init__(
            self,
            damping: float,
            density: Density,
            source: Optional[Sampleable] = None,
        ):
        super().__init__(source)
        self.damping = damping
        self.density = density

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Drift coefficient
        """
        score = self.density.dx_log_density(x)
        return score * self.damping
    
    def noise(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Diffusion coefficient
        """
        return torch.ones_like(x) * math.sqrt(2 * self.damping)

class Hamiltonian(nn.Module, ABC):
    """
    Implementation of Hamiltonian of the form H(x,p) = U(x) + K(x,p)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def hamiltonian(self, x: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def x_dim(self) -> int:
        pass

    def forward(self, xp: Tensor) -> Tensor:
        return self.hamiltonian(xp)

    def dxp_hamiltonian(self, xp):
        xp = xp.unsqueeze(1)
        d_xp = vmap(jacrev(self.hamiltonian, argnums=0))(xp) # (bs, 1, 1, 1, 2 * x_dim)
        return d_xp.squeeze(1,2,3)
    
class StandardHamiltonian(Hamiltonian):
    def __init__(self, density: Density, mass: float):
        super().__init__()
        self.register_module("density", density)
        self.mass = mass

    @property
    def x_dim(self) -> int:
        return self.density.dim

    def hamiltonian(self, xp: Tensor) -> Tensor:
        x, p = xp[:, : self.x_dim], xp[:, self.x_dim :]
        u = - self.density.log_density(x)
        k = 0.5 * torch.sum(p ** 2, dim=1, keepdim=True) / self.mass
        return u + k
    
class ESHHamiltonian(Hamiltonian):
    """
    Based on https://arxiv.org/abs/2111.02434
    """
    def __init__(self, density: Density, mass: float):
        super().__init__()
        self.register_module("density", density)
        self.mass = mass

    @property
    def x_dim(self) -> int:
        return self.density.dim

    def hamiltonian(self, xp: Tensor) -> Tensor:
        x, p = xp[:, : self.x_dim], xp[:, self.x_dim :]

        u = - self.density.log_density(x)
        k = 0.5 * self.mass * torch.log(torch.sum(p ** 2, dim=1, keepdim=True) / self.mass)

        return u + k

class UnderdampedLangevin(ForwardDiffusionProcess):
    def __init__(
        self,
        hamiltonian: Hamiltonian,
        scaling: float,
        damping: float,
        source: Optional[Sampleable] = None,
    ):
        super().__init__(source)
        self.hamiltonian = hamiltonian
        self.x_dim = self.hamiltonian.x_dim
        self.scaling = scaling
        self.damping = damping
        
    def drift(self, x_p: Tensor, t: Tensor) -> Tensor:
        d_xp_h = self.hamiltonian.dxp_hamiltonian(x_p)
        d_x_h = d_xp_h[:, : self.x_dim]
        d_p_h = d_xp_h[:, self.x_dim :]

        x_drift = self.scaling * d_p_h
        p_drift = self.scaling * (- d_x_h - self.damping * d_p_h)

        return torch.cat([x_drift, p_drift], dim=1)
    
    def noise(self, x_p: Tensor, t: Tensor) -> Tensor:
        x, p = x_p[:, : self.x_dim], x_p[:, self.x_dim :]
        x_noise = torch.zeros_like(x)
        p_noise = torch.ones_like(p) * math.sqrt(2 * self.damping * self.scaling)
        return torch.cat([x_noise, p_noise], dim=1)


##############################
# Annealed Langevin Dynamics #
##############################

class AnnealedOverdampedLangevin(ForwardDiffusionProcess):
    """
    Implements a controlled, annealed overdamped Langevin dynamics
    """

    def __init__(
        self,
        control: VectorField,
        damping: float,
        density_path: DensityPath,
        auxiliary_processes: Dict[str, AuxiliaryProcess] = {},
    ):
        super().__init__(density_path.start_sampleable, auxiliary_processes)
        self.register_module("density_path", density_path)
        self.register_module("control", control)
        self.damping = damping

    def drift(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Drift coefficient
        """
        score = self.density_path.dx_log_density(x, t)
        control = self.control(x, t)
        return control + score * self.damping

    def noise(self, x: Tensor, t: Tensor) -> Tensor:
        """Diffusion coefficient"""
        return math.sqrt(2 * self.damping) * torch.ones_like(x)

class AnnealedHamiltonian(nn.Module, ABC):
    """
    Implementation of time-dependent Hamiltonian of the form H_t(x,p) = U_t(x) + K(x,p)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def hamiltonian(self, xp: Tensor, t: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def x_dim(self) -> int:
        pass

    def forward(self, xp: Tensor, t: Tensor) -> Tensor:
        return self.hamiltonian(xp, t)

    def dxp_hamiltonian(self, xp: Tensor, t: Tensor):
        xp = xp.unsqueeze(1)
        t = t.unsqueeze(1) # (bs, 1, 1)
        d_xp = vmap(jacrev(self.hamiltonian, argnums=0))(xp, t) # (bs, 1, 1, 1, 2 * x_dim)
        return d_xp.squeeze(1,2,3)

    @abstractmethod
    def get_start_sampleable(self) -> Sampleable:
        """
        Generate sampleable for the phase-space distribution corresponding to the Hamiltonian at t=0
        """
        pass
    
class AnnealedStandardHamiltonian(AnnealedHamiltonian):
    def __init__(self, density_path: DensityPath, mass: float):
        super().__init__()
        self.register_module("density_path", density_path)
        self.mass = mass

    @property
    def x_dim(self) -> int:
        return self.density_path.dim

    def hamiltonian(self, xp: Tensor, t: Tensor) -> Tensor:
        x, p = xp[:, : self.x_dim], xp[:, self.x_dim :]
        u = - self.density_path.log_density(x, t)
        k = 0.5 * torch.sum(p ** 2, dim=1, keepdim=True) / self.mass
        return u + k
    
    def get_start_sampleable(self) -> Sampleable:
        x_start_sampleable = self.density_path.start_sampleable
        p_start_sampleable = Gaussian.isotropic(dim=self.x_dim, std=math.sqrt(self.mass))
        return ProductSampleable(sampleables=[x_start_sampleable, p_start_sampleable])
        
    
class AnnealedESHHamiltonian(AnnealedHamiltonian):
    """
    Based on https://arxiv.org/abs/2111.02434
    """
    def __init__(self, density_path: DensityPath, mass: float):
        super().__init__()
        self.register_module("density_path", density_path)
        self.mass = mass

    @property
    def x_dim(self) -> int:
        return self.density_path.dim

    def hamiltonian(self, x_p: Tensor, t: Tensor) -> Tensor:
        x, p = x_p[:, : self.x_dim], x_p[:, self.x_dim :]

        u = - self.density_path.log_density(x, t)
        k = 0.5 * self.mass * torch.log(torch.sum(p ** 2, dim=1, keepdim=True) / self.mass)

        return u + k

    def get_start_sampleable(self) -> Sampleable:
        x_start_sampleable = self.density_path.start_sampleable
        p_start_sampleable = Gaussian.isotropic(dim=self.x_dim, std=math.sqrt(self.mass))
        return ProductSampleable(sampleables=[x_start_sampleable, p_start_sampleable])
    
def build_annealed_hamiltonian(density_path: DensityPath, hamiltonian_type: str, mass: float) -> AnnealedHamiltonian:
    if hamiltonian_type == "standard":
        return AnnealedStandardHamiltonian(
            density_path=density_path,
            mass=mass
        )
    elif hamiltonian_type == "esh":
        return AnnealedESHHamiltonian(
            density_path=density_path,
            mass=mass
        )
    else:
        raise ValueError(f"Unknown hamiltonian type: {hamiltonian_type}")


class AnnealedUnderdampedLangevin(ForwardDiffusionProcess):
    """
    Implements underdamped Langevin annealing (i.e., Hamiltonian dynamics)
    """

    def __init__(
        self,
        control: VectorField,
        scaling: float,
        damping: float,
        hamiltonian: AnnealedHamiltonian,
        auxiliary_processes: Dict[str, AuxiliaryProcess] = {},
    ):
        source = hamiltonian.get_start_sampleable()
        super().__init__(source, auxiliary_processes)
        self.register_module("hamiltonian", hamiltonian)
        self.register_module("control", control)
        self.scaling = scaling
        self.damping = damping
        self.x_dim = hamiltonian.x_dim

    def drift(self, x_p: Tensor, t: Tensor) -> Tensor:
        """
        Drift coefficient
        """
        x, _ = x_p[:, : self.x_dim], x_p[:, self.x_dim :]
        d_xp_h = self.hamiltonian.dxp_hamiltonian(x_p, t)
        d_x_h = d_xp_h[:, : self.x_dim]
        d_p_h = d_xp_h[:, self.x_dim :]

        x_drift = self.control(x, t) + self.scaling * d_p_h
        p_drift = self.scaling * (- d_x_h - self.damping * d_p_h)

        return torch.cat([x_drift, p_drift], dim=1)

    def noise(self, x_p: Tensor, t: Tensor) -> Tensor:
        """
        Diffusion coefficient
        """
        x, p = x_p[:, : self.x_dim], x_p[:, self.x_dim :]
        x_noise = torch.zeros_like(x)
        p_noise = torch.ones_like(p) * math.sqrt(2 * self.damping * self.scaling)
        return torch.cat([x_noise, p_noise], dim=1)


###########################################
# Continuously Tempered Langevin Dynamics #
###########################################

class MTUnderdampedSampleable(Sampleable):
    def __init__(
        self,
        xz_source: Sampleable,
        converter: BetaConverter,
        x_mass: float,
        z_mass: float,
    ):
        super().__init__()
        self.register_module("xz_source", xz_source)
        self.converter = converter
        self.x_mass = x_mass
        self.z_mass = z_mass

    @property
    def dim(self) -> int:
        return 2 * self.xz_source.dim

    def sample(self, num_samples: int) -> Tensor:
        xz = self.xz_source.sample(num_samples)
        x, z = xz[:, :-1], xz[:, -1:]
        b = self.converter.xi_to_beta(z)
        px = torch.randn_like(x) * torch.sqrt(self.x_mass / b)
        pz = torch.randn_like(z) * math.sqrt(self.z_mass)
        return torch.cat([xz, px, pz], dim=1)

    
class OverdampedCTDS(ForwardDiffusionProcess):
    pass


class MTAnnealedHamiltonian(nn.Module, ABC):
    """
    Implementation of time-dependent muti-temperature Hamiltonian of the form H_t(xz,pxz) = U_t(xz) + K(xz,pxz)
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def hamiltonian(self, xz_pxz: Tensor, t: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def x_dim(self) -> int:
        pass

    def forward(self, xz_pxz: Tensor, t: Tensor) -> Tensor:
        return self.hamiltonian(xz_pxz, t)

    def d_xz_pxz_hamiltonian(self, xz_pxz: Tensor, t: Tensor):
        xz_pxz = xz_pxz.unsqueeze(1) # (bs, 1, 2 * self.x_dim + 2)
        t = t.unsqueeze(1) # (bs, 1, 1)
        d_xz_pxz = vmap(jacrev(self.hamiltonian, argnums=0))(xz_pxz, t) # (bs, 1, 1, 1, 2 * x_dim + 2)
        return d_xz_pxz.squeeze(1,2,3)

    @abstractmethod
    def get_start_sampleable(self) -> Sampleable:
        """
        Generate sampleable for the phase-space distribution corresponding to the Hamiltonian at t=0
        """
        pass

class StandardMTAnnealedHamiltonian(MTAnnealedHamiltonian):
    def __init__(self, 
                x_mass: float,
                z_mass: float,
                xz_continuum: ReparameterizedMTContinuum,
            ):
        super().__init__()
        self.register_module("xz_continuum", xz_continuum)
        self.x_mass = x_mass
        self.z_mass = z_mass

    @property
    def x_dim(self) -> int:
        return self.xz_continuum.x_dim

    def hamiltonian(self, xz_pxz: Tensor, t: Tensor) -> Tensor:
        # Potential energy
        xz = xz_pxz[:, : self.x_dim + 1]
        u = - self.xz_continuum.log_density(xz, t) # (bs, 1)

        # Kinetic energy
        pxz = xz_pxz[:, self.x_dim + 1 :]
        px, pz = pxz[:, :-1], pxz[:, -1:]

        kx = 0.5 * torch.sum(px ** 2, dim=1, keepdim=True) / self.x_mass # (bs, 1)
        kz = 0.5 * pz ** 2 / self.z_mass # (bs, 1)

        return u + kx + kz
    
    def get_start_sampleable(self) -> Sampleable:
        def sample(num_samples: int) -> Tensor:
            xz = self.xz_continuum.source_sampleable.sample(num_samples)
            x, z = xz[:, :-1], xz[:, -1:]
            px = torch.randn_like(x) * math.sqrt(2 * self.x_mass)
            pz = torch.randn_like(z) * math.sqrt(2 * self.z_mass)
            return torch.cat([xz, px, pz], dim=1)
        return WrapperSampleable(sample_fn=sample, dim=2 * self.x_dim + 2)
    
class TemperedMTAnnealedHamiltonian(MTAnnealedHamiltonian):
    """
    Tempered energy, standard for z
    """
    def __init__(self, 
                x_mass: float,
                z_mass: float,
                xz_continuum: ReparameterizedMTContinuum,
            ):
        super().__init__()
        self.register_module("xz_continuum", xz_continuum)
        self.x_mass = x_mass
        self.z_mass = z_mass

    @property
    def x_dim(self) -> int:
        return self.xz_continuum.x_dim

    def hamiltonian(self, xz_pxz: Tensor, t: Tensor) -> Tensor:
        # Potential energy
        xz = xz_pxz[:, : self.x_dim + 1]
        u = - self.xz_continuum.log_density(xz, t) # (bs, 1)

        # Kinetic energy
        pxz = xz_pxz[:, self.x_dim + 1 :]
        px, pz = pxz[:, :-1], pxz[:, -1:]
        z = xz[:, -1:]
        b = self.xz_continuum.converter.xi_to_beta(z)

        kx = 0.5 * b * torch.sum(px ** 2, dim=1, keepdim=True) / self.x_mass # (bs, 1)
        kz = 0.5 * pz ** 2 / self.z_mass # (bs, 1)

        return u + kx + kz
    
    def get_start_sampleable(self) -> Sampleable:
        def sample(num_samples: int) -> Tensor:
            xz = self.xz_continuum.source_sampleable.sample(num_samples)
            x, z = xz[:, :-1], xz[:, -1:]
            b = self.xz_continuum.converter.xi_to_beta(z)
            px = torch.randn_like(x) * torch.sqrt(2 * self.x_mass / b)
            pz = torch.randn_like(z) * math.sqrt(2 * self.z_mass)
            return torch.cat([xz, px, pz], dim=1)
        return WrapperSampleable(sample_fn=sample, dim=2 * self.x_dim + 2)

class ESHMTAnnealedHamiltonian(MTAnnealedHamiltonian):
    """
    ESH kinetic energy for x (see https://arxiv.org/abs/2111.02434), standard for z
    """
    def __init__(self, 
                x_mass: float,
                z_mass: float,
                xz_continuum: ReparameterizedMTContinuum,
            ):
        super().__init__()
        self.register_module("xz_continuum", xz_continuum)
        self.x_mass = x_mass
        self.z_mass = z_mass

    @property
    def x_dim(self) -> int:
        return self.xz_continuum.x_dim

    def hamiltonian(self, xz_pxz: Tensor, t: Tensor) -> Tensor:
        # Potential energy
        xz = xz_pxz[:, : self.x_dim + 1]
        u = - self.xz_continuum.log_density(xz, t) # (bs, 1)

        # Kinetic energy
        pxz = xz_pxz[:, self.x_dim + 1 :]
        px, pz = pxz[:, :-1], pxz[:, -1:]

        kx = 0.5 * self.x_mass * torch.log(torch.sum(px ** 2, dim=1, keepdim=True) / self.x_mass) # (bs, 1)
        kz = 0.5 * pz ** 2 / self.z_mass # (bs, 1)

        return u + kx + kz
    
    def get_start_sampleable(self) -> Sampleable:
        def sample(num_samples: int) -> Tensor:
            xz = self.xz_continuum.source_sampleable.sample(num_samples)
            x, z = xz[:, :-1], xz[:, -1:]
            px = torch.randn_like(x) * math.sqrt(2 * self.x_mass)
            pz = torch.randn_like(z) * math.sqrt(2 * self.z_mass)
            return torch.cat([xz, px, pz], dim=1)
        return WrapperSampleable(sample_fn=sample, dim=2 * self.x_dim + 2)
    

def build_mt_annealed_hamiltonian(xz_continuum: ReparameterizedMTContinuum, hamiltonian_type: str, x_mass: float, z_mass: float) -> MTAnnealedHamiltonian:
    """
    Factory function to build the appropriate MTAnnealedHamiltonian based on the specified type
    """
    if hamiltonian_type == "standard":
        return StandardMTAnnealedHamiltonian(
            x_mass=x_mass,
            z_mass=z_mass,
            xz_continuum=xz_continuum
        )
    elif hamiltonian_type == "tempered":
        return TemperedMTAnnealedHamiltonian(
            x_mass=x_mass,
            z_mass=z_mass,
            xz_continuum=xz_continuum
        )
    elif hamiltonian_type == "esh":
        return ESHMTAnnealedHamiltonian(
            x_mass=x_mass,
            z_mass=z_mass,
            xz_continuum=xz_continuum
        )
    else:
        raise ValueError(f"Unknown hamiltonian type: {hamiltonian_type}")

class UnderdampedCTDS(ForwardDiffusionProcess):
    """
    CTDS proposal, i.e., a controlled underdamped Langevin dynamics given by the extended Hamiltonian
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
        converter: BetaConverter,
        enforce_z_bounds: bool,
        z_lower_bound: Optional[float] = None,
        z_upper_bound: Optional[float] = None,
        auxiliary_processes: Dict[str, AuxiliaryProcess] = {},
    ):
        super().__init__(source, auxiliary_processes)
        self.register_module("x_control", x_control)
        self.register_module("hamiltonian", hamiltonian)
        self.x_scale = x_scale
        self.z_scale = z_scale
        self.x_damping = x_damping
        self.z_damping = z_damping
        self.x_dim = self.hamiltonian.x_dim
        self.converter = converter
    

        self.enforce_z_bounds = enforce_z_bounds
        self.z_lower_bound = z_lower_bound
        self.z_upper_bound = z_upper_bound
        if self.enforce_z_bounds:
            assert z_lower_bound is not None
            assert z_upper_bound is not None

    
    def drift(self, xz_pxz: Tensor, t: Tensor):
        # Unpack xz_pxz
        xz, pxz = xz_pxz[:, : self.x_dim + 1], xz_pxz[:, self.x_dim + 1 :]
        x, z = xz[:, :-1], xz[:, -1:]

        # Unpack gradient
        d_xz_pxz_h = self.hamiltonian.d_xz_pxz_hamiltonian(xz_pxz, t) # (bs, 2 * x_dim + 2, 2 * x_dim + 2)
        d_x_h = d_xz_pxz_h[:, : self.x_dim]
        d_z_h = d_xz_pxz_h[:, self.x_dim : self.x_dim + 1]
        d_px_h = d_xz_pxz_h[:, self.x_dim + 1 : -1]
        d_pz_h = d_xz_pxz_h[:, -1:]

        # Assemble drifts
        x_drift = self.x_control(x, z, t) + self.x_scale * d_px_h
        px_drift = self.x_scale * (- d_x_h - self.x_damping * d_px_h)

        z_drift = self.z_scale * d_pz_h
        pz_drift = self.z_scale * (- d_z_h - self.z_damping * d_pz_h)

        # Return
        return torch.cat([x_drift, z_drift, px_drift, pz_drift], dim=1)

    def noise(self, xz_pxz: Tensor, t: Tensor):
        """
        Diffusion coefficient
        """
        xz, pxz = xz_pxz[:, : self.x_dim + 1], xz_pxz[:, self.x_dim + 1 :]
        px, pz = pxz[:, :-1], pxz[:, -1:]
        px_noise = torch.ones_like(px) * math.sqrt(2 * self.x_damping * self.x_scale)
        pz_noise = torch.ones_like(pz) * math.sqrt(2 * self.z_damping * self.z_scale)
        return torch.cat(
            [torch.zeros_like(xz), px_noise, pz_noise], dim=1
        )