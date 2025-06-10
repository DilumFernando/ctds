import hydra
from omegaconf import DictConfig

# Temporary debugging tooling
import faulthandler, signal
faulthandler.register(signal.SIGUSR1)

from src.models.ctds import CTDSModule
from src.utils.train import train_module


@hydra.main(version_base=None, config_path="config", config_name="ctds_config")
def main(cfg: DictConfig):
    cfg = cfg.ctds_config
    ctds_module = CTDSModule(cfg)
    breakpoint()

    print(cfg)
    train_module(
        module=ctds_module,
        run_name=cfg.run_name,
        run_group=cfg.run_group,
        max_epochs=cfg.max_epochs,
        val_freq=cfg.val_freq,
        wandb=cfg.wandb,
        wandb_project=cfg.wandb_project,
        num_devices=cfg.num_devices,
        checkpoint=cfg.checkpoint,
        checkpoint_burn_in_epochs=cfg.checkpoint_burn_in_epochs,
    )


if __name__ == "__main__":
    main(None)
