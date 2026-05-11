from __future__ import annotations

import os
from typing import Any


def configure_runtime_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


configure_runtime_env()


def __getattr__(name: str) -> Any:
    if name == "NETSModel":
        from .model import NETSModel

        return NETSModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def train(*args: Any, **kwargs: Any) -> dict:
    from .train import train as train_impl

    return train_impl(*args, **kwargs)


__all__ = ["NETSModel", "train"]
