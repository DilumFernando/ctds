from abc import ABC, abstractmethod
from typing import List

from omegaconf import DictConfig

class AnnealingScheduler(ABC):
    @property
    @abstractmethod
    def T(self) -> float:
        pass

    @abstractmethod
    def step(self):
        pass

    @staticmethod
    def build(cfg: DictConfig) -> "AnnealingScheduler":
        if cfg.annealing_scheduler == "constant":
            return ConstantAnnealingScheduler(cfg.start_T)
        elif cfg.annealing_scheduler == "linear":
            return LinearAnnealingScheduler(cfg.start_T, cfg.dT, cfg.epochs_per_T)
        elif cfg.annealing_scheduler == "exponential":
            return ExponentialAnnealingScheduler(
                cfg.start_T,
                cfg.dT,
                cfg.base_epochs_per_T,
                cfg.epochs_per_T_multiplier,
            )
        elif cfg.annealing_scheduler == "manual":
            return ManualAnnealingScheduler(cfg.T_schedule, cfg.epochs_per_T)
        else:
            raise ValueError("Invalid annealing scheduler")


class ConstantAnnealingScheduler(AnnealingScheduler):
    """
    For use when T is constant (in this case T = 1.0)
    """

    def __init__(self, T: float):
        self._T = T

    @property
    def T(self) -> float:
        return self._T

    def step(self):
        pass


class LinearAnnealingScheduler(AnnealingScheduler):
    def __init__(self, start_T: float, dT: float, epochs_per_T: int):
        self._T = start_T
        self.dT = dT
        self.epochs_per_T = epochs_per_T
        self.epochs_since_update = 0

    @property
    def T(self) -> float:
        return self._T

    def step(self):
        self.epochs_since_update += 1
        if self.epochs_since_update >= self.epochs_per_T:
            if self._T < 1.0:
                self._T += self.dT
            self.epochs_since_update = 0


class ExponentialAnnealingScheduler(AnnealingScheduler):
    """
    Intuition: spend more time closer to T = 1.0
    Explicitly: send base_epochs * multiplier_epochs^T time at each T
    """

    def __init__(
        self,
        start_T: float,
        dT: float,
        base_epochs_per_T: int,
        epochs_per_T_multiplier: float,
    ):
        self._T = start_T
        self.dT = dT
        self.base_epochs_per_T = base_epochs_per_T
        self.epochs_per_T_multiplier = epochs_per_T_multiplier
        self.epochs_per_T = base_epochs_per_T
        self.epochs_since_update = 0

    @property
    def T(self) -> float:
        return self._T

    def step(self):
        self.epochs_since_update += 1
        if self.epochs_since_update >= self.epochs_per_T:
            if self._T < 1.0:
                self._T += self.dT
            self.epochs_since_update = 0
            self.epochs_per_T = int(self.epochs_per_T * self.epochs_per_T_multiplier)


class ManualAnnealingScheduler(AnnealingScheduler):
    def __init__(self, T_schedule: List[float], epochs_per_T: List[float]):
        self.T_schedule = T_schedule
        self.epochs_per_T = epochs_per_T
        self.current_idx = 0
        self.epochs_since_update = 0
        self._T = T_schedule[0]

    @property
    def T(self) -> float:
        return self._T

    def step(self):
        if self.current_idx >= len(self.T_schedule):
            return
        self.epochs_since_update += 1
        if self.epochs_since_update >= self.epochs_per_T[self.current_idx]:
            self.current_idx += 1
            if self.current_idx < len(self.T_schedule):
                self._T = self.T_schedule[self.current_idx]
                self.epochs_since_update = 0
