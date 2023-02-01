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
        self._name = name + f"_x{budget_multiplier}" if budget_multiplier > 1 else name
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
        config_id_to_seeds = self.config_id_to_seeds()
        return Trajectory([
            Entry(
                conf=dict(entry.incumbent),
                loss=entry.train_perf,
                at_iteration=int(entry.ta_runs),
                at_time=entry.wallclock_time,
                seeds=self.get_seeds(entry, config_id_to_seeds)
            )
            for entry in self.initialized_optimizer.get_trajectory()[1:]  # Skip the first one
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

    def get_seeds(self, entry, config_id_to_seeds):
        opt = self.initialized_optimizer
        return config_id_to_seeds[opt.runhistory.config_ids[entry.incumbent]]

    def config_id_to_seeds(self):
        conf_id_to_seeds = defaultdict(list)
        for key in self.initialized_optimizer.runhistory.external.keys():
            conf_id_to_seeds[key.config_id].append(key.seed)
        return conf_id_to_seeds

