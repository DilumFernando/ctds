from __future__ import annotations

import ot
import torch

MAX_LOG_WEIGHT = 4.0


def normalize_log_weights(log_weights: torch.Tensor, log_clamp_val: float = MAX_LOG_WEIGHT) -> torch.Tensor:
    flat_log_weights = log_weights.reshape(-1)
    clamped = torch.clamp(flat_log_weights, max=log_clamp_val)
    shifted = clamped - torch.max(clamped)
    weights = torch.exp(shifted)
    return weights / torch.sum(weights)


def w2_from_samples(
    p_samples: torch.Tensor,
    q_samples: torch.Tensor,
    p_weights: torch.Tensor | None = None,
    q_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    cost_matrix = ot.dist(p_samples, q_samples, metric="sqeuclidean")
    if p_weights is None:
        p_weights = torch.ones_like(p_samples[:, 0]) / p_samples.shape[0]
    if q_weights is None:
        q_weights = torch.ones_like(q_samples[:, 0]) / q_samples.shape[0]
    return ot.emd2(p_weights, q_weights, cost_matrix) ** 0.5


def ess(log_w: torch.Tensor, log_clamp_val: float = MAX_LOG_WEIGHT) -> torch.Tensor:
    w = torch.exp(torch.clamp(log_w, max=log_clamp_val))
    return torch.mean(w) ** 2 / torch.mean(w**2)


def mode_weights_from_samples(
    samples: torch.Tensor,
    modes: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if samples.ndim != 2 or modes.ndim != 2:
        raise ValueError("samples and modes must both have shape [n, dim].")
    distances = torch.cdist(samples, modes)
    assignments = torch.argmin(distances, dim=1)
    if sample_weights is None:
        sample_weights = torch.ones(samples.shape[0], device=samples.device, dtype=samples.dtype)
    else:
        sample_weights = sample_weights.reshape(-1).to(samples)
    sample_weights = sample_weights / torch.sum(sample_weights)
    mode_weights = torch.zeros(modes.shape[0], device=samples.device, dtype=samples.dtype)
    mode_weights.scatter_add_(0, assignments, sample_weights)
    return mode_weights
