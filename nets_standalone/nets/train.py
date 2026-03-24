from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .config import load_config
from .misc import resolve_device
from .model import NETSModel


@dataclass
class CheckpointEntry:
    path: Path
    val_w2: float


def save_checkpoint(path: Path, model: NETSModel, optimizer, scheduler, epoch: int, metrics: Dict[str, float]) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def train(cfg) -> dict:
    device = resolve_device(str(cfg.get("device", "auto")))
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = NETSModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    def lr_lambda(epoch: int) -> float:
        if epoch < int(cfg.lr_burn_in_epochs):
            return 1.0
        step_count = (epoch - int(cfg.lr_burn_in_epochs)) // int(cfg.step_size)
        return float(cfg.gamma) ** step_count

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_id = random.randint(0, 10_000_000)
    run_name = f"{cfg.run_name}_{run_id}"
    checkpoints_dir = Path.cwd() / "checkpoints" / str(cfg.run_group) / run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoints: list[CheckpointEntry] = []
    history = []

    for epoch in range(int(cfg.max_epochs)):
        model.train()
        sample_buffer = None
        if bool(cfg.use_persistent_sample_buffer):
            sample_buffer = model.replenish_sample_buffer(
                num_trajectories=int(cfg.persistent_sample_buffer_trajectories),
                proposal_type="overdamped_langevin",
                T=model.T,
            )

        train_losses = []
        memory_usages = []
        for _ in range(int(cfg.train_steps_per_epoch)):
            optimizer.zero_grad(set_to_none=True)
            start_bytes = torch.cuda.memory_allocated() if device.type == "cuda" else 0
            loss = model.compute_train_loss(sample_buffer)
            loss.backward()
            optimizer.step()
            end_bytes = torch.cuda.memory_allocated() if device.type == "cuda" else 0
            train_losses.append(float(loss.item()))
            memory_usages.append(model.memory_delta_gib(start_bytes, end_bytes))

        scheduler.step()
        model.annealing_scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "memory_gib": float(max(memory_usages) if memory_usages else 0.0),
            "T": float(model.T),
        }

        if (epoch + 1) % int(cfg.val_freq) == 0:
            model.eval()
            val_metrics = model.validate()
            epoch_metrics.update(val_metrics)
            if epoch >= int(cfg.checkpoint_burn_in_epochs):
                checkpoint_path = checkpoints_dir / f"epoch={epoch:03d}-val_w2={val_metrics['val_w2']:.4f}.pt"
                save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, epoch_metrics)
                best_checkpoints.append(CheckpointEntry(checkpoint_path, float(val_metrics["val_w2"])))
                best_checkpoints.sort(key=lambda item: item.val_w2)
                while len(best_checkpoints) > int(cfg.save_top_k):
                    stale = best_checkpoints.pop()
                    if stale.path.exists():
                        stale.path.unlink()

        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics))

    return {"run_name": run_name, "device": str(device), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone NETS with pure PyTorch.")
    parser.add_argument("--config", required=True, help="Path to TOML config.")
    parser.add_argument("--run-name", help="Optional run name override.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Config override, e.g. --set max_epochs=5",
    )
    args = parser.parse_args()

    overrides = list(args.set)
    if args.run_name:
        overrides.append(f"run_name={args.run_name}")
    cfg = load_config(args.config, overrides=overrides)
    train(cfg)


if __name__ == "__main__":
    main()
