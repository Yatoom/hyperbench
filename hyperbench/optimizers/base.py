from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from ConfigSpace import Configuration

from hyperbench.dataset import Dataset
from hyperbench.target_algorithms import TargetAlgorithm
from hyperbench.trajectory.trajectory import Trajectory


class Optimizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def initialize(self, tae_runner: Callable, rng: np.random.RandomState, data: Dataset, budget: int,
                   time_based: bool, target_algorithm: TargetAlgorithm):
        pass

    @abstractmethod
    def search(self) -> Configuration:
        pass

    @abstractmethod
    def get_trajectory(self) -> Trajectory:
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        pass
