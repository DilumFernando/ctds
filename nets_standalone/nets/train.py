from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from .metrics import mode_weights_from_samples
from .model import NETSModel
from .plotting import (
    plotting_available,
    save_metric_history_plots,
    save_weighted_trajectory_plot,
    save_weights_histogram,
)


def resolve_device(device: str | None) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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


def append_jsonl(path: Path, metrics: Dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics) + "\n")


def append_text(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_csv(path: Path, history: list[Dict[str, float]]) -> None:
    if not history:
        return
    fieldnames: list[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


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
    project_root = Path(__file__).resolve().parents[1]
    checkpoints_dir = project_root / "checkpoints" / str(cfg.run_group) / run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    jsonl_log_path = checkpoints_dir / "metrics.jsonl"
    csv_log_path = checkpoints_dir / "metrics.csv"
    text_log_path = checkpoints_dir / "train.log"
    trajectory_plots_dir = checkpoints_dir / "trajectory_plots"
    weight_plots_dir = checkpoints_dir / "weight_plots"
    metric_plots_dir = checkpoints_dir / "metric_plots"

    best_checkpoints: list[CheckpointEntry] = []
    history = []
    wandb_run = None
    plotting_warning_logged = False
    cached_target_plot: dict[str, torch.Tensor] | None = None

    append_text(text_log_path, f"run_name={run_name}")
    append_text(text_log_path, f"device={device}")
    append_text(text_log_path, f"max_epochs={int(cfg.max_epochs)}")

    if bool(cfg.get("wandb", False)):
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Install it or run with wandb=false."
            )
        wandb_project = cfg.get("wandb_project")
        if wandb_project is None:
            raise ValueError("cfg.wandb_project must be set when wandb=true")
        wandb_run = wandb.init(
            project=str(wandb_project),
            group=str(cfg.run_group),
            name=run_name,
            dir=str(checkpoints_dir),
            config=dict(cfg),
        )

    try:
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

            if (
                int(cfg.get("plot_every_n_epochs", 10)) > 0
                and (epoch + 1) % int(cfg.get("plot_every_n_epochs", 10)) == 0
                and int(cfg.x_dim) in (1, 2)
            ):
                if not plotting_available():
                    if not plotting_warning_logged:
                        warning = (
                            "plotting_skipped=matplotlib is not installed; "
                            "install it or set plot_every_n_epochs=0"
                        )
                        append_text(text_log_path, warning)
                        print(warning)
                        plotting_warning_logged = True
                else:
                    plot_buffer = model.sample_plot_buffer(
                        num_trajectories=int(cfg.get("plot_num_trajectories", 32)),
                        min_points=int(cfg.get("plot_min_points", 2000)),
                    )
                    if cached_target_plot is None:
                        cached_target_plot = {
                            "target_samples": plot_buffer["target_samples"],
                            "target_density": plot_buffer["target_density"],
                        }
                        if "target_modes" in plot_buffer:
                            cached_target_plot["target_modes"] = plot_buffer["target_modes"]
                    else:
                        plot_buffer["target_samples"] = cached_target_plot["target_samples"]
                        plot_buffer["target_density"] = cached_target_plot["target_density"]
                        if "target_modes" in cached_target_plot:
                            plot_buffer["target_modes"] = cached_target_plot["target_modes"]
                    if "target_modes" in plot_buffer:
                        particle_mode_weights = mode_weights_from_samples(
                            samples=plot_buffer["xs"][:, -1, :],
                            modes=plot_buffer["target_modes"],
                            sample_weights=plot_buffer["weights"][:, -1, 0],
                        )
                        target_mode_weights = mode_weights_from_samples(
                            samples=plot_buffer["target_samples"],
                            modes=plot_buffer["target_modes"],
                        )
                        for mode_idx, mode_weight in enumerate(particle_mode_weights.tolist()):
                            epoch_metrics[f"particle_mode_{mode_idx}_weight"] = float(mode_weight)
                        for mode_idx, mode_weight in enumerate(target_mode_weights.tolist()):
                            epoch_metrics[f"target_mode_{mode_idx}_weight"] = float(mode_weight)
                    plot_path = trajectory_plots_dir / f"epoch_{epoch + 1:04d}_trajectories.png"
                    hist_path = weight_plots_dir / f"epoch_{epoch + 1:04d}_weights_hist.png"
                    save_weighted_trajectory_plot(
                        xs=plot_buffer["xs"],
                        ts=plot_buffer["ts"],
                        weights=plot_buffer["weights"],
                        target_samples=plot_buffer["target_samples"],
                        target_density=plot_buffer["target_density"],
                        target_modes=plot_buffer.get("target_modes"),
                        particle_mode_weights=particle_mode_weights if "target_modes" in plot_buffer else None,
                        target_mode_weights=target_mode_weights if "target_modes" in plot_buffer else None,
                        path=plot_path,
                        title=f"{run_name} epoch {epoch + 1}",
                    )
                    save_weights_histogram(
                        weights=plot_buffer["weights"],
                        path=hist_path,
                        title=f"{run_name} epoch {epoch + 1} weight histogram",
                    )
                    append_text(text_log_path, f"saved_plot={plot_path}")
                    append_text(text_log_path, f"saved_plot={hist_path}")
                    if wandb_run is not None:
                        wandb.log(
                            {
                                "trajectory_plot": wandb.Image(str(plot_path)),
                                "weights_histogram_plot": wandb.Image(str(hist_path)),
                            },
                            step=epoch,
                        )

            history.append(epoch_metrics)
            append_jsonl(jsonl_log_path, epoch_metrics)
            write_csv(csv_log_path, history)
            append_text(text_log_path, json.dumps(epoch_metrics))

            if wandb_run is not None:
                wandb.log(epoch_metrics, step=epoch)
            print(json.dumps(epoch_metrics))

        if plotting_available():
            save_metric_history_plots(history=history, output_dir=metric_plots_dir)
            append_text(text_log_path, f"saved_metric_plots={metric_plots_dir}")
    finally:
        if wandb_run is not None:
            wandb.finish()

    return {
        "run_name": run_name,
        "device": str(device),
        "history": history,
        "log_dir": str(checkpoints_dir),
        "jsonl_log": str(jsonl_log_path),
        "csv_log": str(csv_log_path),
        "text_log": str(text_log_path),
    }
