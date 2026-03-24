from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig

from ..train import train


@hydra.main(version_base=None, config_path="../conf", config_name="toy_cpu")
def main(cfg: DictConfig) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    train(cfg)


if __name__ == "__main__":
    main()
