import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

import numpy as np
from ConfigSpace import Configuration
from smac.scenario.scenario import Scenario

from hyperbench.api.dataset import Data
from hyperbench.api.target_algorithm import TargetAlgorithm
from hyperbench.api.trajectory import Trajectory


class Optimizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, tae_runner: Callable, rng: np.random.RandomState, data: Data, budget: int,
                   target_algorithm: TargetAlgorithm):
        pass

    @abstractmethod
    def search(self) -> Configuration:
        pass

    @abstractmethod
    def get_trajectory(self) -> Trajectory:
        pass


class SMACBasedOptimizer(Optimizer):

    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.kwargs = kwargs
        self.initialized_optimizer = None

    def initialize(self, tae_runner, rng, data, budget, target_algorithm):
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": budget,
            "ta_run_limit": budget,
            "deterministic": target_algorithm.is_deterministic,
            "cs": target_algorithm.config_space,
            "maxR": 5,
            "output_dir": "./output/smac_output/seed/target_algorithm/task_id",
            "intens_min_chall": 2,
        })

        self.initialized_optimizer = self.optimizer(scenario=scenario, rng=rng, tae_runner=tae_runner, **self.kwargs)
        return self

    def search(self) -> Configuration:
        return self.initialized_optimizer.optimize()

    def get_trajectory(self) -> Trajectory:
        map_id_to_seeds = self.get_trajectory_seeds()
        return Trajectory([
            {
                "conf": entry.incumbent,
                "loss": entry.train_perf,
                "at_iteration": entry.ta_runs,
                "at_time": entry.wallclock_time,
                "seeds": map_id_to_seeds[entry.incumbent_id]
            }
            for entry in self.initialized_optimizer.get_trajectory()
        ])

    def get_trajectory_seeds(self):
        map_id_to_seeds = defaultdict(list)
        for key in self.initialized_optimizer.runhistory.external.keys():
            map_id_to_seeds[key.config_id].append(key.seed)

        return map_id_to_seeds
