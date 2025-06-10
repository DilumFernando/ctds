import unittest
import math

import torch
from omegaconf import DictConfig, OmegaConf

from src.models.ctds import CTDSModule
from src.models.nets import NETSModule, ZeroDtFreeEnergy
from src.simulation.vector_field import (
    LinearGaussianInterpolationVectorField,
)
from src.systems.density_path import LinearGaussianInterpolation
from src.systems.distributions import Gaussian
from src.systems.mt import (
    LinearGaussianContinuum,
    LinearGaussianContinuumVectorField,
    ZeroMTContinuumFreeEnergy,
)
from src.utils.misc import get_device
from src.utils.train import train_module


def dummy_nets_config() -> DictConfig:
    return OmegaConf.create(
        {
            # Run details
            "run_name": "dummy_pinn_test",
            "run_group": "test",
            "wandb": False,

            # Objective details
            "x_dim": 2,
            "target": "fab_gmm",
            "cov_scale": 1.0,
            "density_path": "interpolant",
            "source_std": 2.0,

            # Batching details
            "use_persistent_sample_buffer": False,
            "persistent_sample_buffer_trajectories": 2500,
            "train_batch_size": 20000,
            "train_trajectories": 400,
            "train_steps_per_epoch": 25,
            "val_batch_size": 50000,
            "val_trajectories": 1000,
            "val_steps_per_epoch": 2,
            "val_freq": 5,

            # Over/underdamped proposal details
            "proposal": None,
            "hamiltonian_type": "standard",
            "damping": math.sqrt(8),
            "scale": 1.0,
            "mass": 1.0,
            "use_jarzynski": False,

            # Uniform proposal details
            "uniform_proposal_scales": [10.0, 20.0, 30.0, 40.0, 50.0],
            "uniform_proposal_times": [0.0, 0.25, 0.50, 0.75, 1.0],

            # PINN-specific training details
            "annealing_scheduler": "constant",
            "start_T": 1.0,
            "avg_dt": 0.01,
            "record_every": 2,
            "divergence_mode": "autograd",

            # Debug
            "memory_profile": False,
            "verbose": False,
        }
    )

def train_nets_config() -> DictConfig:
    return OmegaConf.create(
        {
            # Run details
            "run_name": "train_pinn_test",
            "run_group": "test",
            "wandb": False,

            # Objective details
            "x_dim": 2,
            "target": "fab_gmm",
            "cov_scale": 1.0,
            "density_path": "interpolant",
            "source_std": 2.0,

            # Training Details
            "num_devices": 1,
            "max_epochs": 1,
            "val_every_n_epochs": 3,
            "lr": 0.002,
            "lr_burn_in_epochs": 0,
            "step_size": 2,
            "gamma": 0.97,
            "checkpoint_burn_in_epochs": 0,

            # Batching details
            "use_persistent_sample_buffer": True,
            "persistent_sample_buffer_trajectories": 250,
            "train_batch_size": 2000,
            "train_trajectories": 40,
            "train_steps_per_epoch": 25,
            "val_batch_size": 5000,
            "val_trajectories": 100,
            "val_steps_per_epoch": 2,
            "val_freq": 5,

            # Proposal details
            "proposal": "overdamped_langevin",
            "damping": math.sqrt(8),
            "use_jarzynski": False,

            # PINN-specific training details
            "annealing_scheduler": "constant",
            "start_T": 1.0,
            "avg_dt": 0.01,
            "record_every": 2,
            "divergence_mode": "autograd",

            # Control-specific details
            "control": "mlp",
            "control_hiddens": [40, 40, 40, 40],

            # Free-energy specific details
            "free_energy": "mlp",
            "free_energy_hiddens": [40, 40, 40],

            # Time details
            "use_fourier": False,

            # Debug
            "memory_profile": False,
            "verbose": False,
        }
    )


def dummy_ctds_config() -> DictConfig:
    return OmegaConf.create(
        {
            # Run details
            "run_name": "dummy_mt_pinn_test",
            "run_group": "test",
            "wandb": False,

            # Objective details
            "x_dim": 2,
            "mt_continuum": "from_density_path",
            "density_path": "interpolant",
            "target": "fab_gmm",
            "cov_scale": 1.0,
            "source_std": 2.0,

            # Batching details
            "use_persistent_sample_buffer": True,
            "persistent_sample_buffer_trajectories": 2500,
            "train_batch_size": 20000,
            "train_trajectories": 400,
            "train_steps_per_epoch": 25,
            "val_batch_size": 50000,
            "val_trajectories": 1000,
            "val_steps_per_epoch": 2,
            "val_freq": 5,

            # PINN-specific training details
            "annealing_scheduler": "constant",
            "start_T": 1.0,
            "avg_dt": 0.005,
            "record_every": 4,

            # Proposal details
            "proposal": "underdamped_ctds",
            "x_scale": 20.0,
            "z_scale": 5.0,
            "x_damping": 2.0,
            "z_damping": 1.0,
            "hamiltonian_type": "standard",
            "x_mass": 1.0,
            "z_mass": 1.0,
            "beta_min": 0.2,
            "beta_max": 1.0,
            "enforce_z_bounds": False,
            "use_jarzynski": False,
            "divergence_mode": "autograd",

            # Reparameterization details
            "converter": "polynomial",
            "z_lower_bound": -2.0,
            "z_upper_bound": 2.0,
            "delta": 0.25,
            "delta_prime": 1.9,
            "s": 0.8,

            # Confining potential details
            "confining_sharpness": 10.0,
            "confining_lower_bound": -2.0,
            "confining_upper_bound": 2.0,

            # Biasing details
            "biasing_density": "zero",

            # Debug
            "verbose": False,
        }
    )

def train_ctds_config() -> DictConfig:
    return OmegaConf.create(
        {
            # Run details
            "run_name": "train_mt_pinn_test",
            "run_group": "test",
            "wandb": False,

            # Objective details
            "x_dim": 2,
            "mt_continuum": "from_density_path",
            "density_path": "interpolant",
            "target": "fab_gmm",
            "cov_scale": 1.0,
            "source_std": 2.0,

            # Training Details
            "num_devices": 1,
            "max_epochs": 1,
            "val_every_n_epochs": 3,
            "lr": 0.002,
            "lr_burn_in_epochs": 0,
            "step_size": 2,
            "gamma": 0.97,
            "checkpoint_burn_in_epochs": 0,

            # Batching details
            "use_persistent_sample_buffer": True,
            "persistent_sample_buffer_trajectories": 250,
            "train_batch_size": 2000,
            "train_trajectories": 40,
            "train_steps_per_epoch": 25,
            "val_batch_size": 5000,
            "val_trajectories": 100,
            "val_steps_per_epoch": 2,
            "val_freq": 5,

            # PINN-specific training details
            "annealing_scheduler": "constant",
            "start_T": 1.0,
            "avg_dt": 0.005,
            "record_every": 4,

            # Control-specific details
            "x_control": "mlp",
            "x_control_hiddens": [40, 40, 40, 40],

            # Free-energy specific details
            "free_energy": "mlp",
            "free_energy_hiddens": [40, 40, 40],

            # Time details
            "use_fourier": False,

            # Proposal details
            "proposal": "underdamped_ctds",
            "x_scale": 20.0,
            "z_scale": 5.0,
            "x_damping": 2.0,
            "z_damping": 1.0,
            "hamiltonian_type": "standard",
            "x_mass": 1.0,
            "z_mass": 1.0,
            "beta_min": 0.2,
            "beta_max": 1.0,
            "enforce_z_bounds": False,
            "use_jarzynski": False,
            "divergence_mode": "autograd",

            # Reparameterization details
            "converter": "polynomial",
            "z_lower_bound": -2.0,
            "z_upper_bound": 2.0,
            "delta": 0.25,
            "delta_prime": 1.9,
            "s": 0.8,

            # Confining potential details
            "confining_sharpness": 10.0,
            "confining_lower_bound": -2.0,
            "confining_upper_bound": 2.0,

            # Biasing details
            "biasing_density": "zero",

            # Debug
            "verbose": False,
        }
    )


class TestPINN(unittest.TestCase):
    def test_pinn_objective(self):
        cfg = dummy_nets_config()
        device = get_device()
        # Generate a simple density path
        mu_0 = torch.Tensor([-1.0, -2.0])
        sigma_0 = torch.Tensor([[1.0, 1.0], [1.0, 3.0]])
        source = Gaussian(mu_0, sigma_0)

        mu_1 = torch.Tensor([1.0, 3.0])
        sigma_1 = torch.Tensor([[4.0, 0.5], [0.5, 1.0]])
        target = Gaussian(mu_1, sigma_1)

        density_path = LinearGaussianInterpolation(source, target).to(device)

        # Initialize dynamics
        drift = LinearGaussianInterpolationVectorField(source, target).to(device)

        # Initialize PINN module
        test_pinn = NETSModule(
            cfg,
            density_path=density_path,
            control=drift,
            dt_free_energy=ZeroDtFreeEnergy(),
        ).move_to_device(device)

        # Compute loss
        overdamped_loss = test_pinn.pinn_loss_wrapper(
            num_trajectories=cfg.train_trajectories,
            proposal_type="overdamped_langevin",
            T = 1.0,
        )
        underdamped_loss = test_pinn.pinn_loss_wrapper(
            num_trajectories=cfg.train_trajectories,
            proposal_type="underdamped_langevin",
            T = 1.0,
        )
        uniform_loss = test_pinn.pinn_loss_wrapper(
            num_trajectories=cfg.train_trajectories,
            proposal_type="uniform",
            T = 1.0,
        )

        self.assertTrue(overdamped_loss < 1e-5)
        self.assertTrue(underdamped_loss < 1e-5)
        self.assertTrue(uniform_loss < 1e-5)

    def test_pinn_training(self):
        cfg = train_nets_config()
        device = get_device()

        train_pinn = NETSModule(cfg).move_to_device(device)

        train_module(
            module=train_pinn,
            run_name=cfg.run_name,
            run_group=cfg.run_group,
            max_epochs=1,
            wandb=False,
            num_devices=1,
            mute_output=True
        )

    def test_mt_pinn(self):
        cfg = dummy_ctds_config()
        device = get_device()

        mu_0 = torch.Tensor([-1.0, -2.0])
        sigma_0 = torch.Tensor([[1.0, 1.0], [1.0, 3.0]])
        source = Gaussian(mu_0, sigma_0).to(device)

        mu_1 = torch.Tensor([1.0, 3.0])
        sigma_1 = torch.Tensor([[4.0, 0.5], [0.5, 1.0]])
        target = Gaussian(mu_1, sigma_1).to(device)

        mt_continuum = LinearGaussianContinuum(ref_start=source, ref_end=target)
        x_control = LinearGaussianContinuumVectorField(mt_continuum=mt_continuum)
        free_energy = ZeroMTContinuumFreeEnergy()

        mt_pinn_module = CTDSModule(
            cfg=cfg,
            x_control=x_control,
            mt_continuum=mt_continuum,
            free_energy=free_energy,
        ).move_to_device(device)

        # Test PINN loss for each proposal type
        mt_pinn_module.proposal = mt_pinn_module.build_mt_proposal(
            proposal_type="underdamped_ctds",
        ).to(device)
        underdamped_ctds_loss = mt_pinn_module.pinn_loss_wrapper(
            batch_size=cfg.train_batch_size,
            num_trajectories=cfg.train_trajectories,
            T=1.0,
        ).item()

        self.assertTrue(underdamped_ctds_loss < 1e-5)

    def test_mt_pinn_training(self):
        cfg = train_ctds_config()
        device = get_device()

        train_mt_pinn = CTDSModule(cfg).move_to_device(device)

        train_module(
            module=train_mt_pinn,
            run_name=cfg.run_name,
            run_group=cfg.run_group,
            max_epochs=1,
            wandb=False,
            num_devices=1,
            mute_output=True
        )

if __name__ == "__main__":
    unittest.main()
