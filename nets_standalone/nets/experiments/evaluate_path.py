from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .. import plotting as plotting_module
from ..metrics import mode_weights_from_samples, normalize_log_weights, w2_from_samples
from ..model import NETSModel
from ..plotting import (
    plotting_available,
    save_high_dim_cross_section_plots,
    save_high_dim_marginal_histograms,
    save_weighted_trajectory_plot,
    save_weights_histogram,
)
from ..train import resolve_device


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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


def _best_checkpoint(log_dir: Path) -> Path | None:
    checkpoints = list(log_dir.glob("epoch=*-val_w2=*.pt"))
    if not checkpoints:
        return None

    def val_w2(path: Path) -> float:
        try:
            return float(path.stem.split("val_w2=")[-1])
        except ValueError:
            return float("inf")

    return min(checkpoints, key=val_w2)


def _latest_run_dir(cfg: DictConfig) -> Path | None:
    run_group_dir = _project_root() / "checkpoints" / str(cfg.run_group)
    if not run_group_dir.exists():
        return None
    prefix = f"{cfg.run_name}_"
    candidates = [path for path in run_group_dir.iterdir() if path.is_dir() and path.name.startswith(prefix)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_checkpoint(cfg: DictConfig) -> Path:
    explicit_path = cfg.get("eval_checkpoint_path")
    if explicit_path:
        checkpoint = Path(str(explicit_path)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        if checkpoint.is_dir():
            best = _best_checkpoint(checkpoint)
            if best is None:
                raise FileNotFoundError(f"No epoch=*-val_w2=*.pt checkpoint found in {checkpoint}")
            return best
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    run_dir = _latest_run_dir(cfg)
    if run_dir is None:
        raise FileNotFoundError(
            "Could not infer a run directory. Pass +eval_checkpoint_path=/path/to/checkpoint.pt "
            "or +eval_checkpoint_path=/path/to/run_dir."
        )
    checkpoint = _best_checkpoint(run_dir)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No epoch=*-val_w2=*.pt checkpoint found in {run_dir}. "
            "Set save_top_k > 0 during training or pass +eval_checkpoint_path explicitly."
        )
    return checkpoint


def _ess_count(log_weights: torch.Tensor) -> float:
    weights = normalize_log_weights(log_weights)
    if weights.numel() == 0:
        return float("nan")
    return float((1.0 / torch.sum(weights.square()).clamp_min(torch.finfo(weights.dtype).tiny)).item())


def _logmeanexp(values: torch.Tensor) -> float:
    values = values.reshape(-1)
    return float((torch.logsumexp(values, dim=0) - math.log(max(values.numel(), 1))).item())


def _mmd_rbf(samples: torch.Tensor, target_samples: torch.Tensor, max_points: int = 1000) -> float:
    x = samples
    y = target_samples.to(samples)
    if x.shape[0] > max_points:
        x = x[torch.randperm(x.shape[0], device=x.device)[:max_points]]
    if y.shape[0] > max_points:
        y = y[torch.randperm(y.shape[0], device=y.device)[:max_points]]

    with torch.no_grad():
        dxx = torch.cdist(x, x).square()
        dyy = torch.cdist(y, y).square()
        dxy = torch.cdist(x, y).square()
        median_sq = torch.median(dxy.detach()).clamp_min(torch.finfo(x.dtype).eps)
        gamma = 1.0 / (2.0 * median_sq)
        mmd = torch.exp(-gamma * dxx).mean() + torch.exp(-gamma * dyy).mean() - 2.0 * torch.exp(-gamma * dxy).mean()
    return float(torch.clamp(mmd, min=0.0).item())


def _dim_pairs(value: Any, dim: int) -> list[tuple[int, int]]:
    if value is not None:
        pairs = []
        for pair in value:
            if isinstance(pair, str):
                left, right = pair.split(",", maxsplit=1)
                pairs.append((int(left), int(right)))
            else:
                if len(pair) != 2:
                    raise ValueError("eval_dim_pairs entries must have exactly two dimensions.")
                pairs.append((int(pair[0]), int(pair[1])))
        return pairs

    max_dims = min(dim, 8)
    pairs = [(idx, idx + 1) for idx in range(0, max_dims - 1, 2)]
    return pairs or [(0, 0)]


def _path_metrics(
    model: NETSModel,
    sample_buffer: dict[str, torch.Tensor],
    target_samples: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    xs = sample_buffer["xs"]
    ts = sample_buffer["ts"]
    log_weights = sample_buffer["log_weights"]
    target_modes = getattr(model.density_path.end_sampleable, "means", None)
    target_modes = None if target_modes is None else target_modes.to(xs)

    for time_idx in range(xs.shape[1]):
        samples = xs[:, time_idx, :]
        t = ts[:, time_idx, :]
        t_value = float(t.mean().item())
        step_log_weights = log_weights[:, time_idx, :]
        normalized_weights = normalize_log_weights(step_log_weights).to(samples)
        path_log_density = model.density_path.log_density(samples, t)
        log_density_ratio = path_log_density - step_log_weights

        row: dict[str, Any] = {
            "step": time_idx,
            "t": t_value,
            "elbo": float(torch.mean(log_density_ratio).item()),
            "eubo": _logmeanexp(log_density_ratio),
            "ess_count": _ess_count(step_log_weights),
            "w2": float("nan"),
            "weighted_w2": float("nan"),
            "mmd": float("nan"),
        }

        try:
            row["w2"] = float(w2_from_samples(samples, target_samples.to(samples)).item())
            target_weights = torch.ones(target_samples.shape[0], device=samples.device) / target_samples.shape[0]
            row["weighted_w2"] = float(
                w2_from_samples(
                    samples,
                    target_samples.to(samples),
                    p_weights=normalized_weights,
                    q_weights=target_weights,
                ).item()
            )
        except Exception as exc:
            row["w2_error"] = str(exc)
            row["weighted_w2_error"] = str(exc)

        row["mmd"] = _mmd_rbf(samples, target_samples.to(samples))

        if target_modes is not None:
            mode_weights = mode_weights_from_samples(
                samples=samples,
                modes=target_modes,
                sample_weights=sample_buffer["weights"][:, time_idx, 0],
            )
            for mode_idx, mode_weight in enumerate(mode_weights.tolist()):
                row[f"mode_weight_{mode_idx}"] = float(mode_weight)

        rows.append(row)

    return rows


def _plot_path_metrics(metrics_history: list[dict[str, Any]], output_dir: Path) -> None:
    if not plotting_available() or not metrics_history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = plotting_module.plt
    if plt is None:
        return

    steps = [int(row["step"]) for row in metrics_history]
    metric_keys = [
        key
        for key in metrics_history[0]
        if key != "step" and not key.startswith("mode_weight_") and not key.endswith("_error")
    ]
    for key in metric_keys:
        values = [row.get(key, float("nan")) for row in metrics_history]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(steps, values, linewidth=2.0)
        ax.scatter(steps, values, s=18)
        ax.set_title(key)
        ax.set_xlabel("path step")
        ax.set_ylabel(key)
        ax.grid(alpha=0.35, linestyle="--", linewidth=0.8, color="#666666")
        fig.tight_layout()
        fig.savefig(output_dir / f"{key}.png", dpi=180)
        plt.close(fig)

    mode_keys = sorted(
        [key for key in metrics_history[0] if key.startswith("mode_weight_")],
        key=lambda key: int(key.split("_")[-1]),
    )
    if mode_keys:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key in mode_keys:
            ax.plot(steps, [row.get(key, float("nan")) for row in metrics_history], linewidth=2.0, label=key)
        ax.set_title("mode_weights")
        ax.set_xlabel("path step")
        ax.set_ylabel("mode weight")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.35, linestyle="--", linewidth=0.8, color="#666666")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "mode_weights.png", dpi=180)
        plt.close(fig)


def _save_eval_sample_plots(
    cfg: DictConfig,
    sample_buffer: dict[str, torch.Tensor],
    target_samples: torch.Tensor,
    output_dir: Path,
    title: str,
) -> dict[str, Any]:
    if not plotting_available():
        return {}

    target_log_density = cfg_model_density.end_sampleable.log_density(target_samples)
    target_modes = getattr(cfg_model_density.end_sampleable, "means", None)
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}

    if int(cfg.x_dim) in (1, 2):
        samples_path = output_dir / "samples.png"
        weights_path = output_dir / "weights_hist.png"
        save_weighted_trajectory_plot(
            xs=sample_buffer["xs"],
            ts=sample_buffer["ts"],
            weights=sample_buffer["weights"],
            target_samples=target_samples,
            target_density=torch.exp(target_log_density).detach(),
            target_modes=None if target_modes is None else target_modes.detach(),
            particle_mode_weights=None,
            target_mode_weights=None,
            path=samples_path,
            title=title,
        )
        save_weights_histogram(
            weights=sample_buffer["weights"],
            path=weights_path,
            title=f"{title} weight histogram",
        )
        result["samples_plot"] = str(samples_path)
        result["weights_plot"] = str(weights_path)
    else:
        dim_pairs = _dim_pairs(cfg.get("eval_dim_pairs"), int(cfg.x_dim))
        cross_section_paths = save_high_dim_cross_section_plots(
            xs=sample_buffer["xs"],
            weights=sample_buffer["weights"],
            target_samples=target_samples,
            target_modes=None if target_modes is None else target_modes.detach(),
            output_dir=output_dir / "cross_sections",
            title=title,
            dim_pairs=dim_pairs,
            max_pairs=int(cfg.get("eval_cross_section_max_pairs", 6)),
        )
        marginal_path = save_high_dim_marginal_histograms(
            xs=sample_buffer["xs"],
            weights=sample_buffer["weights"],
            target_samples=target_samples,
            output_dir=output_dir / "marginals",
            title=f"{title} marginal histograms",
            dims=None if cfg.get("eval_marginal_dims") is None else [int(x) for x in cfg.get("eval_marginal_dims")],
            max_dims=int(cfg.get("eval_marginal_max_dims", 8)),
        )
        result["cross_section_plots"] = ";".join(str(path) for path in cross_section_paths)
        result["marginal_plot"] = str(marginal_path)

    return result


cfg_model_density = None


@hydra.main(version_base=None, config_path="../conf", config_name="learnable_warmstart_prior_constrained_modes_2d_fixed_beta")
def main(cfg: DictConfig) -> None:
    global cfg_model_density

    device = resolve_device(str(cfg.get("device", "auto")))
    checkpoint_path = _resolve_checkpoint(cfg)
    model = NETSModel(cfg).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()
    cfg_model_density = model.density_path

    output_root = Path(cfg.get("eval_output_dir") or checkpoint_path.parent / "eval" / "path")
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        sample_buffer = model.replenish_sample_buffer(
            num_trajectories=int(cfg.get("eval_num_trajectories", cfg.get("val_trajectories", 1000))),
            proposal_type=str(cfg.get("eval_proposal", cfg.proposal)),
            T=float(cfg.get("eval_T", 1.0)),
        )
        target_samples = model.density_path.end_sampleable.sample(
            int(cfg.get("eval_target_samples", cfg.get("val_trajectories", 1000)))
        ).to(device)
        metrics_history = _path_metrics(model, sample_buffer, target_samples)

    csv_path = metrics_dir / "path_metrics.csv"
    jsonl_path = metrics_dir / "path_metrics.jsonl"
    pth_path = metrics_dir / "path_metrics_history.pth"
    _write_csv(csv_path, metrics_history)
    _write_jsonl(jsonl_path, metrics_history)
    torch.save(metrics_history, pth_path)
    torch.save(sample_buffer["xs"].detach().cpu(), metrics_dir / "generated_trajectory.pth")
    torch.save(sample_buffer["log_weights"].detach().cpu(), metrics_dir / "generated_log_weights.pth")
    torch.save(sample_buffer["weights"].detach().cpu(), metrics_dir / "generated_weights.pth")
    torch.save(target_samples.detach().cpu(), metrics_dir / "target_samples.pth")

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "output_root": str(output_root),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "num_trajectories": int(sample_buffer["xs"].shape[0]),
        "trajectory_length": int(sample_buffer["xs"].shape[1]),
    }
    torch.save(metadata, metrics_dir / "eval_metadata.pth")
    with (metrics_dir / "eval_metadata.txt").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2) + "\n")

    _plot_path_metrics(metrics_history, plots_dir)
    plot_artifacts = _save_eval_sample_plots(
        cfg=cfg,
        sample_buffer=sample_buffer,
        target_samples=target_samples,
        output_dir=plots_dir / "samples",
        title=f"{cfg.run_name} path eval",
    )

    final_metrics = metrics_history[-1] if metrics_history else {}
    torch.save(final_metrics, metrics_dir / "final_metrics.pth")
    print(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "metrics_csv": str(csv_path),
                "metrics_jsonl": str(jsonl_path),
                "plots_dir": str(plots_dir),
                "final_metrics": final_metrics,
                **plot_artifacts,
            }
        )
    )


if __name__ == "__main__":
    main()
