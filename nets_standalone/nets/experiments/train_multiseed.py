from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ..model import NETSModel
from ..plotting import (
    plotting_available,
    save_high_dim_cross_section_plots,
    save_high_dim_marginal_histograms,
    save_weighted_trajectory_plot,
    save_weights_histogram,
)
from ..train import resolve_device, train


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _copy_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _cfg_list(value: Any) -> list:
    if value is None:
        return []
    return list(value)


def _dim_pairs(value: Any) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    pairs = []
    for pair in value:
        if len(pair) != 2:
            raise ValueError("cross_section_dim_pairs entries must have exactly two dimensions.")
        pairs.append((int(pair[0]), int(pair[1])))
    return pairs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _best_checkpoint(log_dir: str | Path) -> Path | None:
    checkpoints = list(Path(log_dir).glob("epoch=*-val_w2=*.pt"))
    if not checkpoints:
        return None

    def val_w2(path: Path) -> float:
        try:
            return float(path.stem.split("val_w2=")[-1])
        except ValueError:
            return float("inf")

    return min(checkpoints, key=val_w2)


def _numeric_summary(rows: list[dict[str, Any]]) -> list[dict[str, float | str]]:
    if not rows:
        return []
    keys = sorted({key for row in rows for key in row})
    summary = []
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            summary.append(
                {
                    "metric": key,
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n": float(len(values)),
                }
            )
    return summary


def _evaluate_and_plot(cfg: DictConfig, checkpoint_path: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    device = resolve_device(str(cfg.get("device", "auto")))
    model = NETSModel(cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()

    eval_metrics = model.validate()
    eval_metrics = {f"eval_{key[4:]}" if key.startswith("val_") else f"eval_{key}": value for key, value in eval_metrics.items()}

    if not plotting_available() or int(cfg.get("eval_plot_every_seed", 1)) <= 0:
        return eval_metrics

    plot_dir = output_dir / f"seed_{seed}" / "eval_plots"
    plot_buffer = model.sample_plot_buffer(
        num_trajectories=int(cfg.get("eval_plot_num_trajectories", cfg.get("plot_num_trajectories", 64))),
        min_points=int(cfg.get("eval_plot_min_points", cfg.get("plot_min_points", 2000))),
    )
    title = f"{cfg.run_name} seed {seed} eval"

    if int(cfg.x_dim) in (1, 2):
        trajectory_path = plot_dir / "samples.png"
        weights_path = plot_dir / "weights_hist.png"
        save_weighted_trajectory_plot(
            xs=plot_buffer["xs"],
            ts=plot_buffer["ts"],
            weights=plot_buffer["weights"],
            target_samples=plot_buffer["target_samples"],
            target_density=plot_buffer["target_density"],
            target_modes=plot_buffer.get("target_modes"),
            particle_mode_weights=None,
            target_mode_weights=None,
            path=trajectory_path,
            title=title,
        )
        save_weights_histogram(
            weights=plot_buffer["weights"],
            path=weights_path,
            title=f"{title} weight histogram",
        )
        eval_metrics["eval_samples_plot"] = str(trajectory_path)
        eval_metrics["eval_weights_plot"] = str(weights_path)
    else:
        cross_section_paths = save_high_dim_cross_section_plots(
            xs=plot_buffer["xs"],
            weights=plot_buffer["weights"],
            target_samples=plot_buffer["target_samples"],
            target_modes=plot_buffer.get("target_modes"),
            output_dir=plot_dir / "cross_sections",
            title=title,
            dim_pairs=_dim_pairs(cfg.get("cross_section_dim_pairs")),
            max_pairs=int(cfg.get("cross_section_max_pairs", 6)),
        )
        marginal_path = save_high_dim_marginal_histograms(
            xs=plot_buffer["xs"],
            weights=plot_buffer["weights"],
            target_samples=plot_buffer["target_samples"],
            output_dir=plot_dir / "marginals",
            title=f"{title} marginal histograms",
            dims=None if cfg.get("marginal_dims") is None else [int(x) for x in cfg.get("marginal_dims")],
            max_dims=int(cfg.get("marginal_max_dims", 8)),
        )
        eval_metrics["eval_cross_section_plots"] = ";".join(str(path) for path in cross_section_paths)
        eval_metrics["eval_marginal_plot"] = str(marginal_path)

    return eval_metrics


@hydra.main(version_base=None, config_path="../conf", config_name="constrained_random_modes_2d")
def main(cfg: DictConfig) -> None:
    seeds = _cfg_list(cfg.get("multiseed_seeds"))
    if not seeds:
        num_seeds = int(cfg.get("num_seeds", 1))
        start_seed = int(cfg.get("seed", 0))
        seeds = list(range(start_seed, start_seed + num_seeds))
    seeds = [int(seed) for seed in seeds]

    base_run_name = str(cfg.run_name)
    base_run_group = str(cfg.run_group)
    aggregate_dir = _project_root() / "checkpoints" / base_run_group / f"{base_run_name}_multiseed"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        seed_cfg = _copy_cfg(cfg)
        seed_cfg.seed = seed
        seed_cfg.run_name = f"{base_run_name}_seed_{seed}"
        if bool(seed_cfg.get("multiseed_vary_target_seed", False)):
            seed_cfg.target_seed = seed

        result = train(seed_cfg)
        row: dict[str, Any] = {
            "seed": seed,
            "run_name": result["run_name"],
            "log_dir": result["log_dir"],
        }
        if result["history"]:
            row.update({f"final_{key}": value for key, value in result["history"][-1].items()})

        checkpoint_path = _best_checkpoint(result["log_dir"])
        if checkpoint_path is not None:
            row["best_checkpoint"] = str(checkpoint_path)
            row.update(_evaluate_and_plot(seed_cfg, checkpoint_path, aggregate_dir, seed))
        else:
            row["best_checkpoint"] = ""
            row["eval_warning"] = "No checkpoint found; set save_top_k > 0 and checkpoint_burn_in_epochs < max_epochs."

        rows.append(row)
        _write_jsonl(aggregate_dir / "multiseed_metrics.jsonl", rows)
        _write_csv(aggregate_dir / "multiseed_metrics.csv", rows)
        _write_csv(aggregate_dir / "multiseed_summary.csv", _numeric_summary(rows))

    print(json.dumps({"aggregate_dir": str(aggregate_dir), "num_seeds": len(seeds)}))


if __name__ == "__main__":
    main()
