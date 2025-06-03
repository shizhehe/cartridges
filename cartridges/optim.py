"""
The code in this module is adapted from the "Olmo" training library. 
https://github.com/allenai/OLMo/blob/main/olmo/train.py
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from math import cos, pi
from typing import Optional

from pydrantic import ObjectConfig

from capsules.utils import get_logger


log = get_logger(__name__)


class Scheduler(metaclass=ABCMeta):
    class Config(ObjectConfig):
        _pass_as_config = True

        # This maximum number of steps is used to calculate the schedule, it is not
        # used to limit the number of steps that training runs for. Once the steps
        # exceed this, the schedule will simply use the final learning rate.
        max_steps: Optional[int]

        # The starting point for the warmup
        warmup_min_lr: Optional[float] = 5e-3

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        raise NotImplementedError

    def _linear_warmup(self, initial_lr: float, step: int, warmup_steps: int = 2000) -> float:
        warmup_min_lr = self.config.warmup_min_lr if self.config.warmup_min_lr is not None else initial_lr * 0.10
        assert 0 <= warmup_min_lr < initial_lr
        return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps


@dataclass
class CosWithWarmup(Scheduler):

    class Config(Scheduler.Config):
        warmup_steps: int
        alpha_f: float = 0.1
    
    def __init__(self, config: Config):
        self.config = config

    def get_lr(self, initial_lr: float, step: int) -> float:
        max_steps = self.config.max_steps
        eta_min = initial_lr * self.config.alpha_f
        if step < self.config.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.config.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.config.warmup_steps
            max_steps = max_steps - self.config.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


@dataclass
class LinearWithWarmup(Scheduler):
    class Config(Scheduler.Config):
        warmup_steps: int
        alpha_f: float = 0.1
    
    def __init__(self, config: Config):
        self.config = config

    def get_lr(self, initial_lr: float, step: int) -> float:
        max_steps = self.config.max_steps
        eta_min = initial_lr * self.config.alpha_f
        if step < self.config.warmup_steps:
            return self._linear_warmup(initial_lr, step, self.config.warmup_steps)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.config.warmup_steps
            max_steps = max_steps - self.config.warmup_steps
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)




