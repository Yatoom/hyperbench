from collections import defaultdict

import numpy as np
from ConfigSpace import Configuration
from smac.scenario.scenario import Scenario

from hyperbench.optimizers.base import Optimizer
from hyperbench.trajectory.entry import Entry
from hyperbench.trajectory.trajectory import Trajectory


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

    def initialize(self, tae_runner, seed, data, budget, time_based, target_algorithm):
        rng = np.random.RandomState(seed)
        scenario = Scenario({
            "run_obj": "quality",
            "ta_run_limit": budget * self.budget_multiplier if not time_based else "inf",
            "wallclock_limit": budget * self.budget_multiplier if time_based else "inf",
            "deterministic": target_algorithm.deterministic,
            "cs": target_algorithm.config_space(),
            "maxR": 5,
            "output_dir": f"./smac_output/{target_algorithm.name}/{self.name}/{seed}/{data.metadata.name}",
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
