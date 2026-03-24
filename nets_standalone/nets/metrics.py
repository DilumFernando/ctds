from __future__ import annotations

import ot
import torch

MAX_LOG_WEIGHT = 4.0


def w2_from_samples(p_samples: torch.Tensor, q_samples: torch.Tensor) -> torch.Tensor:
    cost_matrix = ot.dist(p_samples, q_samples, metric="sqeuclidean")
    p_weights = torch.ones_like(p_samples[:, 0]) / p_samples.shape[0]
    q_weights = torch.ones_like(q_samples[:, 0]) / q_samples.shape[0]
    return ot.emd2(p_weights, q_weights, cost_matrix) ** 0.5


def ess(log_w: torch.Tensor, log_clamp_val: float = MAX_LOG_WEIGHT) -> torch.Tensor:
    w = torch.exp(torch.clamp(log_w, max=log_clamp_val))
    return torch.mean(w) ** 2 / torch.mean(w**2)
