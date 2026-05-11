from __future__ import annotations

import ot
import torch

MAX_LOG_WEIGHT = 4.0
MIN_LOG_WEIGHT = -80.0


def sanitize_log_weights(log_weights: torch.Tensor, log_clamp_val: float = MAX_LOG_WEIGHT) -> torch.Tensor:
    return torch.nan_to_num(
        log_weights,
        nan=MIN_LOG_WEIGHT,
        posinf=log_clamp_val,
        neginf=MIN_LOG_WEIGHT,
    ).clamp(min=MIN_LOG_WEIGHT, max=log_clamp_val)


def normalize_log_weights(log_weights: torch.Tensor, log_clamp_val: float = MAX_LOG_WEIGHT) -> torch.Tensor:
    flat_log_weights = sanitize_log_weights(log_weights, log_clamp_val=log_clamp_val).reshape(-1)
    if flat_log_weights.numel() == 0:
        return flat_log_weights
    clamped = torch.clamp(flat_log_weights, max=log_clamp_val)
    shifted = clamped - torch.max(clamped)
    weights = torch.exp(shifted)
    total = torch.sum(weights).clamp_min(torch.finfo(weights.dtype).tiny)
    return weights / total


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
    w = torch.exp(sanitize_log_weights(log_w, log_clamp_val=log_clamp_val))
    denom = torch.mean(w**2).clamp_min(torch.finfo(w.dtype).tiny)
    return torch.mean(w) ** 2 / denom


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
    sample_weights = torch.nan_to_num(sample_weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    weight_sum = torch.sum(sample_weights)
    if weight_sum <= 0:
        sample_weights = torch.ones_like(sample_weights) / sample_weights.numel()
    else:
        sample_weights = sample_weights / weight_sum
    mode_weights = torch.zeros(modes.shape[0], device=samples.device, dtype=samples.dtype)
    mode_weights.scatter_add_(0, assignments, sample_weights)
    return mode_weights
