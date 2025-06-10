from typing import Callable, Optional, Tuple

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import Tensor

from src.models.ctds import MTPINNProposal
from src.models.nets import PINNProposal
from src.simulation.dynamics import ForwardProcess
from src.systems.base import Sampleable
from src.systems.mt import BetaConverter
from src.utils.misc import get_device, record_every_idxs, get_module_device

#######################################
# High-Dimensional Plotting Functions #
#######################################


@torch.no_grad()
def kdeplot_sampleable(
    sampleable: Sampleable,
    num_samples: int,
    ax: Optional[Axes] = None,
    dims: Tuple[int, int] = (0, 1),
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples)  # (ns, dim)
    sns.kdeplot(
        x=samples[:, dims[0]].detach().cpu(),
        y=samples[:, dims[1]].detach().cpu(),
        ax=ax,
        **kwargs,
    )
    return samples


@torch.no_grad()
def scatter_sampleable(
    sampleable: Sampleable,
    num_samples: int,
    ax: Optional[Axes] = None,
    dims: Tuple[int, int] = (0, 1),
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples)  # (ns, dim)
    ax.scatter(
        samples[:, dims[0]].detach().cpu(),
        samples[:, dims[1]].detach().cpu(),
        **kwargs,
    )
    return samples


######################################
# Two-Dimensional Plotting Functions #
######################################


@torch.no_grad()
def plot_proposal(
    proposal: PINNProposal,
    num_samples: int,
    ts: Tensor,
    axes=None,
    scale=10.0,
    use_alpha=True,
    marker="x",
    color="black",
    s=15,
    dims: Tuple[int, int] = (0, 1),
    mode: str = "scatter",
    **kwargs,
) -> Tuple[plt.Axes, Tensor]:
    """
    Plot samples from the proposal at given timesteps. Note that the returned ts array might have fewer timesteps than the input ts due to only plotting every k-th timestep (see 'record_every' parameter to proposal)
    Args:
    - ts: (nts,)
    Returns:
    - axes: used for plotting
    - ts: (nts, 1) (pruned to only those corresponding to plotted samples

    """
    ts = ts.view(1, -1, 1).expand(num_samples, -1, 1)
    trajectory, ts, log_weights = proposal.sample(
        ts
    )  # (bs, nts, data_dim), (bs, nts, 1), (bs, nts, 1)
    weights = torch.exp(log_weights)  # (bs, nts, 1)
    max_weight = torch.max(weights, dim=0, keepdim=True).values  # (1, nts, 1)
    min_weight = torch.min(weights, dim=0, keepdim=True).values  # (1, nts, 1)
    eps = 1e-5
    weights = (weights - min_weight + eps) / (
        max_weight - min_weight + eps
    )  # (bs, nts, 1)
    _, nts, _ = ts.shape

    if axes is None:
        _, axes = plt.subplots(1, nts, figsize=(12 * nts, 12))
    for t_idx in range(nts):
        alpha = weights[:, t_idx].detach().cpu().numpy() if use_alpha else 1.0
        if mode == "scatter":
            axes[t_idx].scatter(
                trajectory[:, t_idx, dims[0]].detach().cpu(),
                trajectory[:, t_idx, dims[1]].detach().cpu(),
                alpha=alpha,
                marker=marker,
                color=color,
                s=s,
                **kwargs,
            )
        elif mode == "kdeplot":
            sns.kdeplot(
                x=trajectory[:, t_idx, dims[0]].detach().cpu(),
                y=trajectory[:, t_idx, dims[1]].detach().cpu(),
                ax=axes[t_idx],
                **kwargs,
            )
        axes[t_idx].set_xlim(-scale, scale)
        axes[t_idx].set_ylim(-scale, scale)
        t = ts[0, t_idx].item()
        axes[t_idx].set_title(f"t = {t:.2f}")
    return axes, ts[0]

@torch.no_grad()
def mt_plot_proposal(
    proposal: MTPINNProposal,
    num_samples: int,
    nts: Tensor,
    t_bins: Tensor,
    b_bins: Tensor,
    axes=None,
    scale=10.0,
    marker="x",
    color="black",
    s=15,
    dims: Tuple[int, int] = (0, 1),
    **kwargs,
) -> Tuple[plt.Axes, Tensor]:
    """
    Plot samples from the proposal at given timesteps. Note that the returned ts array might have fewer timesteps than the input ts due to only plotting every k-th timestep (see 'record_every' parameter to proposal)
    Args:
    - ts: (nts,)
    - beta_bins: (nbs,)
    Returns:
    - axes: used for plotting
    - ts: (nts, 1) (pruned to only those corresponding to plotted samples
    """
    device = get_module_device(proposal)
    ts = torch.linspace(0, 1, nts).view(1, -1, 1).expand(num_samples, -1, 1).to(device)
    sample_results = proposal.sample(ts, **kwargs)
    sample_ts = sample_results["ts"]  # (bs * nts, 1)
    sample_bs = sample_results["bs"]  # (bs * nts, 1)
    sample_xs = sample_results["xs"]  # (bs * nts, dim)

    if axes is None:
        _, axes = plt.subplots(len(b_bins), len(t_bins), figsize=(8 * len(t_bins), 8 * len(b_bins)))
    axes = axes.reshape((len(b_bins), len(t_bins)))

    for t_idx in range(len(t_bins)):
        t_mask = torch.argmin(torch.abs(sample_ts - t_bins.view(1, -1)), dim=1) == t_idx  # (bs * nts,)
        for b_idx in range(len(b_bins)):
            b_mask = torch.argmin(torch.abs(sample_bs - b_bins.view(1, -1)), dim=1) == b_idx  # (bs * nts,)
            mask = torch.logical_and(t_mask, b_mask)  # (bs * nts,)
            xs_to_plot = sample_xs[mask]

            # Plot samples
            ax = axes[b_idx, t_idx]
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            t = t_bins[t_idx].item()
            b = b_bins[b_idx].item()
            nxs = xs_to_plot.shape[0]
            ax.set_title(f"{nxs} samples at: t = {t:.2f}, beta = {b:.2f}")
            ax.scatter(
                xs_to_plot[:, dims[0]].cpu(),
                xs_to_plot[:, dims[1]].cpu(),
                marker=marker,
                color=color,
                s=s,
            )
    return sample_results, axes

@torch.no_grad()
def old_mt_plot_proposal(
    proposal: MTPINNProposal,
    num_samples: int,
    ts: Tensor,
    beta_bins: Tensor,
    axes=None,
    scale=10.0,
    use_alpha=True,
    marker="x",
    color="black",
    s=15,
    dims: Tuple[int, int] = (0, 1),
    **kwargs,
) -> Tuple[plt.Axes, Tensor]:
    """
    Plot samples from the proposal at given timesteps. Note that the returned ts array might have fewer timesteps than the input ts due to only plotting every k-th timestep (see 'record_every' parameter to proposal)
    Args:
    - ts: (nts,)
    - beta_bins: (nbs,)
    Returns:
    - axes: used for plotting
    - ts: (nts, 1) (pruned to only those corresponding to plotted samples

    """
    ts = ts.view(1, -1, 1).expand(num_samples, -1, 1)
    sample_results = proposal.sample(ts, **kwargs)
    plotted_trajectory = sample_results["xbs"]  # (bs, nts, dim+1)
    ts = sample_results["ts"]  # (bs, nts, 1)
    weights = sample_results["weights"]  # (bs, nts, 1)

    ts = ts[0]  # (nts, 1)
    nts = len(ts)
    nbs = len(beta_bins)

    if axes is None:
        _, axes = plt.subplots(nbs, nts, figsize=(8 * nts, 8 * nbs))
    axes = axes.reshape((nbs, nts))
    plotted_trajectory = plotted_trajectory.detach()
    for t_idx in range(nts):
        xb_at_t = plotted_trajectory[:, t_idx]  # (bs, dim+1)
        beta_at_t = xb_at_t[:, -1:]  # (bs, 1)
        beta_dists = torch.abs(
            beta_at_t.view(-1, 1) - beta_bins.view(1, -1)
        )  # (bs, nbs)
        sample_bins = torch.argmin(beta_dists, dim=1)  # (bs,)
        for b_idx in range(nbs):
            # Select all samples in bin b_idx
            samples_to_plot = sample_bins == b_idx
            xbs = xb_at_t[samples_to_plot]  # (nxs, dim)
            ax = axes[b_idx, t_idx]
            if len(xbs) > 0:
                bin_weights = weights[samples_to_plot, t_idx, 0]
                bin_weights /= torch.max(bin_weights)
                # bin_weights = torch.pow(bin_weights, 0.5)
                alpha = bin_weights.detach().cpu() if use_alpha else 1.0
                ax.scatter(
                    xbs[:, dims[0]].cpu(),
                    xbs[:, dims[1]].cpu(),
                    alpha=alpha,
                    marker=marker,
                    color=color,
                    s=s,
                )
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            t = ts[t_idx].item()
            b = beta_bins[b_idx].item()
            nxs = xbs.shape[0]
            ax.set_title(f"{nxs} samples at: t = {t:.2f}, beta = {b:.2f}")
    return ts, axes


@torch.no_grad()
def plot_process(
    process: ForwardProcess,
    num_samples: int,
    timesteps: Tensor,
    record_every: int = 1,
    axes=None,
    scale=10.0,
    marker="x",
    color="black",
    s=15,
    dims: Tuple[int, int] = (0, 1),
) -> Tuple[plt.Axes, Tensor]:
    """
    Plot samples from the diffusion process at given timesteps
    Args:
    - timesteps: (num_timesteps,)
    Returns:
    - axes: used for plotting
    - plotted_timesteps: pruned to only those corresponding to plotted samples
    """
    timesteps = timesteps.view(1, -1, 1).expand(num_samples, -1, 1)  # (bs, nts, 1)
    plotted_trajectory, _, plotted_ts = process.sample_with_trajectory(
        num_samples=num_samples,
        ts=timesteps,
        record_every=record_every,
        use_tqdm=True
    )  # (bs, nts, data_dim)
    timesteps = timesteps[0]  # (num_timesteps,1)
    plotted_ts = plotted_ts[0]  # (num_timesteps,1)

    num_plots = plotted_trajectory.shape[1]
    if axes is None:
        _, axes = plt.subplots(1, num_plots, figsize=(12 * num_plots, 12))
    plotted_trajectory = plotted_trajectory.detach().cpu().numpy()
    for t_idx in range(num_plots):
        axes[t_idx].scatter(
            plotted_trajectory[:, t_idx, dims[0]],
            plotted_trajectory[:, t_idx, dims[1]],
            alpha=0.5,
            marker=marker,
            color=color,
            s=s,
        )
        axes[t_idx].set_xlim(-scale, scale)
        axes[t_idx].set_ylim(-scale, scale)
        t = plotted_ts[t_idx].item()
        axes[t_idx].set_title(f"t = {t:.2f}")
    return axes, plotted_ts

@torch.no_grad()
def mt_plot_mixed_process(
    process: ForwardProcess,
    converter: BetaConverter,
    num_samples: int,
    timesteps: torch.Tensor,
    betas: torch.Tensor,
    record_every: int = 1,
    axes=None,
    scale=10.0,
    marker="x",
    color="black",
    dims: Tuple[int, int] = (0, 1),
    s=15,
):
    """
    Used to plot time evolution of 2D processes over (x,xi).
    Betas arg used for bins rather than directly for sampling.
    Args:
    - timesteps: (nts,)
    - betas: (nbs,) # Used for binning
    """
    # Process inputs
    timesteps = timesteps.clone()
    timesteps = timesteps.view(1, -1, 1).expand(num_samples, -1, 1)  # (bs, nts, 1)

    plotted_trajectory_xz_pxzs, _, plotted_ts = process.sample_with_trajectory(
        ts=timesteps, num_samples=num_samples, record_every=record_every, use_tqdm=True
    )  # (bs, nts, data_dim + 1)
    # Remove any extra dimensions such as momenta
    plotted_trajectory_xzs = plotted_trajectory_xz_pxzs[:, :, :3]
    # Convert from xi to beta
    zs = plotted_trajectory_xzs[:, :, -1:]  # (bs, nts, 1)
    bs = converter.xi_to_beta(zs)  # (bs, nts, 1)
    plotted_trajectory = torch.cat([plotted_trajectory_xzs[:, :, :-1], bs], dim=-1)

    plotted_ts = plotted_ts[0]  # (nts, 1)
    nts, nbs = len(plotted_ts), len(betas)

    if axes is None:
        _, axes = plt.subplots(nbs, nts, figsize=(8 * nts, 8 * nbs))
    axes = axes.reshape((nbs, nts))
    plotted_trajectory = plotted_trajectory.detach()
    for t_idx in range(nts):
        xb_at_t = plotted_trajectory[:, t_idx]  # (bs, dim+1)
        x_at_t = xb_at_t[:, :-1]  # (bs, dim)
        beta_at_t = xb_at_t[:, -1:]  # (bs, 1)
        beta_dists = torch.abs(beta_at_t.view(-1, 1, 1) - betas.view(1, -1, 1))
        beta_bins = torch.argmin(beta_dists, dim=1)
        for b_idx in range(nbs):
            # Select all samples in bin b_idx
            bin_idxs = torch.where(beta_bins == b_idx)[0]
            xs = xb_at_t[bin_idxs]  # (nxs, dim)
            ax = axes[b_idx, t_idx]
            ax.scatter(
                xs[:, dims[0]].cpu(),
                xs[:, dims[1]].cpu(),
                alpha=0.5,
                marker=marker,
                color=color,
                s=s,
            )
            ax.set_xlim(-scale, scale)
            ax.set_ylim(-scale, scale)
            t = plotted_ts[t_idx].item()
            b = betas[b_idx].item()
            nxs = xs.shape[0]
            ax.set_title(f"{nxs} samples at: t = {t:.2f}, beta = {b:.2f}")
    return plotted_trajectory, axes, plotted_ts, plotted_trajectory_xz_pxzs


@torch.no_grad()
def plot_vector_field(
    vector_field: Callable[[Tensor], Tensor],
    ax=None,
    scale: float = 1.0,
    num_points: int = 100,
    device: Optional[torch.device] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot gradients of fn over [-scale, scale] x [-scale, scale]
    """
    if ax is None:
        _, ax = plt.subplots()
    if device is None:
        device = get_device()
    query_points = torch.linspace(-scale, scale, num_points).to(device)
    x, y = torch.meshgrid(query_points, query_points)
    z = vector_field(torch.stack([x, y], dim=-1).reshape(-1, 2))
    ax.quiver(x.detach().cpu(), y.detach().cpu(), z[:, 0].detach().cpu(), z[:, 1].detach().cpu(), **kwargs)
    return ax


@torch.no_grad()
def plot_contours(
    fn: Callable[[Tensor], Tensor],
    ax=None,
    scale: float = 1.0,
    num_points: int = 100,
    min_value: float = -1e4,
    legend: bool = True,
    device: Optional[torch.device] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot contours of fn over [-scale, scale] x [-scale, scale]
    """
    if device is None:
        device = get_device()
    if ax is None:
        _, ax = plt.subplots()
    query_points = torch.linspace(-scale, scale, num_points).to(device)
    x, y = torch.meshgrid(query_points, query_points)
    z = fn(torch.stack([x, y], dim=-1).reshape(-1, 2)).reshape(num_points, num_points)
    z = torch.clamp_min(z, min_value).detach().cpu().numpy()
    x = x.detach().cpu()
    y = y.detach().cpu()
    contour = ax.contour(
        x,
        y,
        z,
        extents=[-scale, scale, -scale, scale],
        origin="lower",
        colors="grey",
        linestyles="solid",
        **kwargs,
    )
    if legend:
        ax.legend(*contour.legend_elements(), loc="upper right")
    return ax


def plot_density(
    fn: Callable[[Tensor], Tensor],
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    bins: int = 250,
    scale: int = 5.0,
    device: Optional[torch.device] = None,
    cmap: str = "Blues",
    **kwargs,
) -> plt.Axes:
    if device is None:
        device = get_device()
    if ax is None:
        fig, ax = plt.subplots()
    x = torch.linspace(-scale, scale, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y)
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    density = fn(xy).detach().cpu().numpy().reshape(bins, bins).T
    im = ax.imshow(
        density,
        extent=[-scale, scale, -scale, scale],
        origin="lower",
        cmap=cmap,
        **kwargs,
    )
    if fig:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")
    return ax


# Extend above plotting functions to time-dependent fns
def make_timeseries_plottable(plottable: Callable) -> Callable:
    def timeseries_plottable(
        fn_x_t: Callable[[Tensor, Tensor], Tensor],
        ts: torch.Tensor,
        axes=None,
        **kwargs,
    ):
        """
        Args:
        - ts: (nts, 1)
        - xs: (bs, nts, dim)
        """
        if axes is None:
            _, axes = plt.subplots(1, len(ts), figsize=(4 * len(ts), 4))
        for t_idx in range(len(ts)):
            ax = axes[t_idx]
            t = ts[t_idx : t_idx + 1]

            def fn_x(xt):
                return fn_x_t(xt, t.view(1, 1).expand(xt.shape[0], 1))

            plottable(fn_x, ax=ax, **kwargs)

    return timeseries_plottable


timeseries_plot_vector_field = make_timeseries_plottable(plot_vector_field)
timeseries_plot_contours = make_timeseries_plottable(plot_contours)
timeseries_plot_density = make_timeseries_plottable(plot_density)


def make_mt_timeseries_plottable(plottable: Callable) -> Callable:
    def mt_timeseries_plottable(
        fn_xb_t: Callable[[Tensor, Tensor], Tensor],
        ts: Tensor,
        bs: Tensor,
        axes=None,
        **kwargs,
    ):
        """
        Args:
        - ts: (nts, 1)
        - bs: (nbs, 1)
        """
        if axes is None:
            _, axes = plt.subplots(len(bs), len(ts), figsize=(8 * len(ts), 8 * len(bs)))
            axes = axes.reshape((len(bs), len(ts)))
        for b_idx in range(len(bs)):
            for t_idx in range(len(ts)):
                ax = axes[b_idx, t_idx]
                t = ts[t_idx : t_idx + 1]
                b = bs[b_idx : b_idx + 1]

                def fn_x(x):
                    # Concatenate x and b to form state
                    _xb = torch.cat([x, b.view(1, 1).expand(x.shape[0], 1)], dim=1)
                    _t = t.view(1, 1).expand(x.shape[0], 1)
                    return fn_xb_t(_xb, _t)

                plottable(fn_x, ax=ax, **kwargs)

    return mt_timeseries_plottable


mt_timeseries_plot_vector_field = make_mt_timeseries_plottable(plot_vector_field)
mt_timeseries_plot_contours = make_mt_timeseries_plottable(plot_contours)
mt_timeseries_plot_density = make_mt_timeseries_plottable(plot_density)
