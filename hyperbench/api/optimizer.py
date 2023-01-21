import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable

import numpy as np
from ConfigSpace import Configuration
from smac.scenario.scenario import Scenario

from hyperbench.api.dataset import Data
from hyperbench.api.target_algorithm import TargetAlgorithm
from hyperbench.api.trajectory import Trajectory, Entry


class Optimizer(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self):
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

    @abstractmethod
    def get_stats(self) -> dict:
        pass


class SMACBasedOptimizer(Optimizer):

    def __init__(self, optimizer, name, budget_multiplier=1, **kwargs):
        self.optimizer = optimizer
        self._name = name
        self.kwargs = kwargs
        self.initialized_optimizer = None
        self.budget_multiplier = budget_multiplier

    @property
    def name(self):
        return self._name

    def initialize(self, tae_runner, seed, data, budget, target_algorithm):
        rng = np.random.RandomState(seed)
        scenario = Scenario({
            "run_obj": "quality",
            "runcount-limit": budget * self.budget_multiplier,
            "ta_run_limit": budget * self.budget_multiplier,
            "deterministic": target_algorithm.is_deterministic,
            "cs": target_algorithm.config_space,
            "maxR": 5,
            "output_dir": f"./smac_output/{target_algorithm.name}/{self.name}/{seed}/{data.parent.name}",
            "intens_min_chall": 2,
        })

        self.initialized_optimizer = self.optimizer(scenario=scenario, rng=rng, tae_runner=tae_runner, **self.kwargs)
        return self

    def search(self) -> Configuration:
        return self.initialized_optimizer.optimize()

    def get_trajectory(self) -> Trajectory:
        map_id_to_seeds = self.get_trajectory_seeds()
        return Trajectory([
            Entry(
                conf=dict(entry.incumbent),
                loss=entry.train_perf,
                at_iteration=int(np.ceil(entry.ta_runs / self.budget_multiplier)),
                at_time=entry.wallclock_time / self.budget_multiplier,
                seeds=map_id_to_seeds[entry.incumbent_id]
            )
            for entry in self.initialized_optimizer.get_trajectory()
        ])

    def get_stats(self):
        stats = self.initialized_optimizer.stats
        return {
            "mean_cost": np.mean([i.cost for i in self.initialized_optimizer.runhistory.data.values()]),
            "finished_ta_runs": stats.finished_ta_runs,
            "inc_changed": stats.inc_changed,
            "n_configs": stats.n_configs,
            "submitted_ta_runs": stats.submitted_ta_runs,
            "ta_time_used": stats.ta_time_used,
            "wallclock_time_used": stats.wallclock_time_used
        }

    def get_trajectory_seeds(self):
        map_id_to_seeds = defaultdict(list)
        for key in self.initialized_optimizer.runhistory.external.keys():
            map_id_to_seeds[key.config_id].append(key.seed)

        return map_id_to_seeds
