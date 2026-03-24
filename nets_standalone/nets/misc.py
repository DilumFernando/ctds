from __future__ import annotations

from functools import wraps
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

BYTES_PER_GIB = 1024 * 1024 * 1024


def resolve_device(device: str | None) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def cuda_profile(fn):
    @wraps(fn)
    def wrapper(*args, profile: bool = False, **kwargs):
        if profile and torch.cuda.is_available():
            start_bytes = torch.cuda.memory_allocated()
            result = fn(*args, **kwargs)
            end_bytes = torch.cuda.memory_allocated()
            byte_diff_gib = (end_bytes - start_bytes) / BYTES_PER_GIB
            print(f"Call to {fn.__name__} used {byte_diff_gib:.3f} GiB of memory")
            return result
        return fn(*args, **kwargs)

    return wrapper


def get_module_device(module: nn.Module) -> Optional[torch.device]:
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        pass
    return None


def record_every_idxs(num_timesteps: int, record_every: int) -> Tensor:
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [torch.arange(0, num_timesteps - 1, record_every), torch.tensor([num_timesteps - 1])]
    )
