from typing import Final, Optional

import ot
import torch

from src.systems.base import Sampleable

MAX_LOG_WEIGHT: Final[float] = 4.0


def mmd(p: Sampleable, q: Sampleable, num_samples: int = 2000) -> torch.Tensor:
    raise NotImplementedError("MMD is not implemented yet.")


W2_INF: Final[float] = 5000
W2_MIN_SAMPLES: Final[int] = 50


def w2_from_samples(p_samples: torch.Tensor, q_samples: torch.Tensor) -> torch.Tensor:
    """
    Compute the Wasserstein-2 distance between two sets of samples.
    Args:
        p_samples: Samples from the first distribution, shape (num_samples, dim).
        q_samples: Samples from the second distribution, shape (num_samples, dim).
    Returns:
        torch.Tensor: Wasserstein-2 distance.
    """
    cost_matrix = ot.dist(p_samples, q_samples, metric="sqeuclidean")
    p_weights = torch.ones_like(p_samples[:, 0]) / p_samples.shape[0]  # (num_samples,)
    q_weights = torch.ones_like(q_samples[:, 0]) / q_samples.shape[0]  # (num_samples,)
    return ot.emd2(p_weights, q_weights, cost_matrix) ** 0.5


def w2(p: Sampleable, q: Sampleable, num_samples: int = 2000) -> torch.Tensor:
    """Wasserstein-2 distance between two distributions.

    Args:
        p (Sampleable): First distribution.
        q (Sampleable): Second distribution.

    Returns:
        Wasserstein-2 distance: (1,).
    """
    p_samples = p.sample(num_samples)  # (num_samples, dim)
    q_samples = q.sample(num_samples)  # (num_samples, dim)
    return w2_from_samples(p_samples, q_samples)


def ess(log_w: torch.Tensor, log_clamp_val: Optional[float] = None) -> torch.Tensor:
    """Effective Sample Size (ESS) of a set of weights.

    Args:
        log_w (torch.Tensor): Log weights of shape (N,).

    Returns:
        torch.Tensor: ESS of the weights.
    """
    if log_clamp_val is None:
        log_clamp_val = MAX_LOG_WEIGHT
    w = torch.exp(torch.clamp(log_w, max=log_clamp_val))
    return torch.mean(w) ** 2 / torch.mean(w**2)
