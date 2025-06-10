from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

BYTES_PER_MIB = 1024 * 1024
BYTES_PER_GIB = 1024 * 1024 * 1024


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_allocated_bytes(device: torch.device, marker: str) -> None:
    """
    Print the number of bytes allocated on the device
    """
    bytes_allocated = torch.cuda.memory_allocated(device)
    print(
        f"Bytes allocated at {marker}: {bytes_allocated}, MiB: {bytes_allocated / BYTES_PER_MIB}, GiB: {bytes_allocated / BYTES_PER_GIB}"
    )


def cuda_profile(fn):
    """
    Measure the memory usage of a function
    """

    def wrapper(*args, profile: bool = False, **kwargs):
        if profile:
            start_bytes = torch.cuda.memory_allocated()
            result = fn(*args, **kwargs)
            end_bytes = torch.cuda.memory_allocated()
            byte_diff_gib = (end_bytes - start_bytes) / BYTES_PER_GIB
            print(f"Call to {fn.__name__} used {byte_diff_gib:.3f} GiB of memory")
            return result
        else:
            return fn(*args, **kwargs)

    return wrapper


def get_module_device(module: nn.Module) -> Optional[torch.device]:
    """
    Grab the device of the first parameter or buffer in the module
    """
    if not isinstance(module, nn.Module):
        raise ValueError("Expected a nn.Module, got {}".format(type(module)))
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        pass
    # In the case that the module has no parameters or buffers, but still produces output (e.g., a random number generator)
    if hasattr(module, "module_device"):
        return module.module_device
    return None


def record_every_idxs(num_timesteps: int, record_every: int) -> Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )
