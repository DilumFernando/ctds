from __future__ import annotations

from pathlib import Path

import torch

PLOT_LIMIT = 20.0

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError:
    matplotlib = None
    plt = None
    Normalize = None


def plotting_available() -> bool:
    return matplotlib is not None and plt is not None and Normalize is not None


def save_weighted_trajectory_plot(
    xs: torch.Tensor,
    ts: torch.Tensor,
    weights: torch.Tensor,
    target_samples: torch.Tensor,
    target_density: torch.Tensor,
    target_modes: torch.Tensor | None,
    particle_mode_weights: torch.Tensor | None,
    target_mode_weights: torch.Tensor | None,
    path: str | Path,
    title: str,
) -> None:
    if not plotting_available():
        raise ImportError("matplotlib is not installed. Install it or disable plotting.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs_cpu = xs.detach().cpu()
    ts_cpu = ts.detach().cpu()
    final_xs_cpu = xs_cpu[:, -1]
    final_weights_cpu = weights.detach().cpu()[:, -1].reshape(-1)
    target_samples_cpu = target_samples.detach().cpu()
    target_density_cpu = target_density.detach().cpu().reshape(-1)

    x_dim = xs.shape[-1]

    if x_dim not in (1, 2):
        raise ValueError("Trajectory plotting is only supported for x_dim in {1, 2}.")

    if x_dim == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        left_ax, right_ax = axes
    else:
        fig, left_ax = plt.subplots(1, 1, figsize=(7, 5.5))
        right_ax = None

    weights_norm = Normalize(vmin=float(final_weights_cpu.min()), vmax=float(final_weights_cpu.max()))
    density_norm = Normalize(vmin=float(target_density_cpu.min()), vmax=float(target_density_cpu.max()))
    target_modes_cpu = None if target_modes is None else target_modes.detach().cpu()

    if x_dim == 2:
        particle_scatter = left_ax.scatter(
            final_xs_cpu[:, 0],
            final_xs_cpu[:, 1],
            c=final_weights_cpu.numpy(),
            cmap="magma",
            norm=weights_norm,
            s=18,
            alpha=0.9,
        )
        target_scatter = right_ax.scatter(
            target_samples_cpu[:, 0],
            target_samples_cpu[:, 1],
            c=target_density_cpu.numpy(),
            cmap="magma",
            norm=density_norm,
            s=22,
            alpha=0.9,
        )
        left_ax.set_xlabel("x")
        left_ax.set_ylabel("y")
        left_ax.set_aspect("equal", adjustable="box")
        right_ax.set_xlabel("x")
        right_ax.set_ylabel("y")
        right_ax.set_aspect("equal", adjustable="box")
        if target_modes_cpu is not None:
            right_ax.scatter(
                target_modes_cpu[:, 0],
                target_modes_cpu[:, 1],
                marker="x",
                s=180,
                linewidths=4.0,
                color="black",
                alpha=0.95,
                zorder=5,
            )
            right_ax.scatter(
                target_modes_cpu[:, 0],
                target_modes_cpu[:, 1],
                marker="x",
                s=120,
                linewidths=2.6,
                color="#00e5ff",
                alpha=0.98,
                zorder=6,
            )

        shared_xlim = (-PLOT_LIMIT, PLOT_LIMIT)
        shared_ylim = (-PLOT_LIMIT, PLOT_LIMIT)
        left_ax.set_xlim(shared_xlim)
        right_ax.set_xlim(shared_xlim)
        left_ax.set_ylim(shared_ylim)
        right_ax.set_ylim(shared_ylim)
    else:
        particle_values = final_xs_cpu[:, 0].numpy()
        particle_weights = final_weights_cpu.numpy()
        target_values = target_samples_cpu[:, 0].reshape(-1).numpy()

        left_ax.hist(
            target_values,
            bins=60,
            range=(-PLOT_LIMIT, PLOT_LIMIT),
            density=True,
            color="#f28e2b",
            alpha=0.35,
            edgecolor="white",
            linewidth=0.4,
            label="target samples",
        )
        left_ax.hist(
            particle_values,
            bins=60,
            range=(-PLOT_LIMIT, PLOT_LIMIT),
            density=True,
            color="#808080",
            alpha=0.35,
            edgecolor="white",
            linewidth=0.4,
            label="particle samples",
        )
        left_ax.hist(
            particle_values,
            bins=60,
            range=(-PLOT_LIMIT, PLOT_LIMIT),
            weights=particle_weights,
            density=True,
            color="#2ca02c",
            alpha=0.9,
            edgecolor="white",
            linewidth=0.4,
            label="weighted particles",
        )
        left_ax.set_xlabel("x")
        left_ax.set_ylabel("density")
        if target_modes_cpu is not None:
            left_ax.scatter(
                target_modes_cpu[:, 0],
                torch.zeros_like(target_modes_cpu[:, 0]),
                marker="x",
                s=180,
                linewidths=4.0,
                color="black",
                alpha=0.95,
                zorder=5,
            )
            left_ax.scatter(
                target_modes_cpu[:, 0],
                torch.zeros_like(target_modes_cpu[:, 0]),
                marker="x",
                s=120,
                linewidths=2.6,
                color="#00e5ff",
                alpha=0.98,
                zorder=6,
            )

        shared_x_range = (-PLOT_LIMIT, PLOT_LIMIT)
        left_ax.set_xlim(shared_x_range)
        left_ax.set_ylim(bottom=0.0)
        left_ax.legend(loc="upper right")

    if x_dim == 2:
        left_ax.set_title("Particles colored by weights")
        right_ax.set_title("Target samples colored by density")
    else:
        left_ax.set_title("1D distribution comparison")
    fig.suptitle(title)
    left_ax.grid(alpha=0.45, linestyle="--", linewidth=0.9, color="#666666")
    left_ax.set_box_aspect(1)
    if right_ax is not None:
        right_ax.grid(alpha=0.45, linestyle="--", linewidth=0.9, color="#666666")
        right_ax.set_box_aspect(1)

    if x_dim == 2:
        left_ax.text(
            0.01,
            0.01,
            f"final time t={float(ts_cpu[:, -1].max()):.2f}",
            transform=left_ax.transAxes,
            fontsize=9,
            alpha=0.8,
        )

        particle_cbar = fig.colorbar(particle_scatter, ax=left_ax)
        particle_cbar.set_label("particle weight")
        target_cbar = fig.colorbar(target_scatter, ax=right_ax)
        target_cbar.set_label("target density")

    if particle_mode_weights is not None:
        particle_summary = ", ".join(
            f"m{idx}={float(weight):.3f}" for idx, weight in enumerate(particle_mode_weights)
        )
        left_ax.text(
            0.02,
            0.98,
            f"particle mode weights: {particle_summary}",
            transform=left_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "#666666"},
        )
    if target_mode_weights is not None and right_ax is not None:
        target_summary = ", ".join(
            f"m{idx}={float(weight):.3f}" for idx, weight in enumerate(target_mode_weights)
        )
        right_ax.text(
            0.02,
            0.98,
            f"target mode weights: {target_summary}",
            transform=right_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.82, "edgecolor": "#666666"},
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_weights_histogram(
    weights: torch.Tensor,
    path: str | Path,
    title: str,
    bins: int = 30,
) -> None:
    if not plotting_available():
        raise ImportError("matplotlib is not installed. Install it or disable plotting.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    weights_cpu = weights.detach().cpu().reshape(-1).numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(weights_cpu, bins=bins, color="#2a9d8f", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_metric_history_plots(history: list[dict], output_dir: str | Path) -> None:
    if not plotting_available():
        raise ImportError("matplotlib is not installed. Install it or disable plotting.")
    if not history:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = []
    for row in history:
        for key in row.keys():
            if key != "epoch" and key not in metric_keys:
                metric_keys.append(key)

    epochs = [row["epoch"] for row in history]
    for metric_key in metric_keys:
        ys = []
        xs = []
        for row in history:
            value = row.get(metric_key)
            if value is None:
                continue
            xs.append(row["epoch"])
            ys.append(value)
        if not ys:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(xs, ys, color="#1f77b4", linewidth=2.0)
        ax.scatter(xs, ys, color="#1f77b4", s=18)
        ax.set_title(metric_key.replace("_", " "))
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric_key)
        ax.grid(alpha=0.35, linestyle="--", linewidth=0.8, color="#666666")
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric_key}.png", dpi=180)
        plt.close(fig)
