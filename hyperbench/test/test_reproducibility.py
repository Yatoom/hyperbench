import dataclasses
import os
import unittest

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from smac.facade.roar_facade import ROAR
from smac.facade.smac_hpo_facade import SMAC4HPO

from hyperbench.benchmark import BenchmarkConfig
from hyperbench.benchmark import BenchmarkRunner
from hyperbench.optimizers import SMACBasedOptimizer
from hyperbench.provider import OpenMLProvider
from hyperbench.target_algorithms import RandomForest
from hyperbench.trajectory import Trajectory
from hyperbench.transformer import SimpleTransformer


class TestReproducibility(unittest.TestCase):
    benchmark = BenchmarkConfig(
        budget=150,
        time_based=False,
        transformer=SimpleTransformer(),
        output_folder=f"/tmp/results/",
        scoring="balanced_accuracy",
        seeds=[2268061101],
        target_algorithms=[RandomForest()],
        datasets=[OpenMLProvider(11)],
        optimizers=[
            SMACBasedOptimizer(ROAR, "roar_x2", budget_multiplier=2),
            SMACBasedOptimizer(SMAC4HPO, "smac", budget_multiplier=1)
        ],
        search_eval_splits=StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0),
        train_test_splits=ShuffleSplit(n_splits=3, random_state=0, test_size=0.10),
    )

    def setUp(self):
        for i in [0, 1]:
            folder = os.path.join(self.benchmark.output_folder, str(i))
            bench = dataclasses.replace(self.benchmark, output_folder=folder)
            BenchmarkRunner(bench).start()

    def compare(self, optimizer, file):
        target_algorithm = self.benchmark.target_algorithms[0].name
        seed = str(self.benchmark.seeds[0])
        dataset = self.benchmark.datasets[0].data.metadata.name

        first = os.path.join(self.benchmark.output_folder, str(0), target_algorithm, optimizer, seed, dataset, file)
        second = os.path.join(self.benchmark.output_folder, str(1), target_algorithm, optimizer, seed, dataset, file)
        first_trajectory = Trajectory.load(first)
        second_trajectory = Trajectory.load(second)
        self.assertEqual(first_trajectory.as_list, second_trajectory.as_list)

    def test_roar_evaluations_equal(self):
        self.compare("roar_x2", "eval.json")

    def test_roar_searches_equal(self):
        self.compare("roar_x2", "eval.json")

    def test_smac_evaluations_equal(self):
        self.compare("smac", "search.json")

    def test_smac_searches_equal(self):
        self.compare("smac", "search.json")
