from __future__ import annotations

from nets import configure_runtime_env

configure_runtime_env()

import hydra
from omegaconf import DictConfig

from ..train import train


@hydra.main(version_base=None, config_path="../conf", config_name="toy_cpu")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
