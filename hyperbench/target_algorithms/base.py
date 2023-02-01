import dataclasses
from abc import ABC, abstractmethod

from ConfigSpace import ConfigurationSpace, Configuration
import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

from hyperbench.dataset import Dataset
from hyperbench.dataset.metadata import Metadata
from hyperbench.trajectory import Trajectory


class BaseTarget(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def deterministic(self):
        pass

    @staticmethod
    @abstractmethod
    def init_model(seed, metadata: Metadata, **config) -> any:
        pass

    @staticmethod
    @abstractmethod
    def constants() -> dict:
        pass

    @staticmethod
    def config_space() -> ConfigurationSpace:
        pass

    def get_config_evaluator(self, dataset: Dataset, train_test_splits, scoring, progress, loop_iterations):
        def evaluate(config: Configuration, seed: int):
            # Initialize algorithm
            algorithm = self.init_model(seed, dataset.metadata, **dict(config))

            # Perform cross validation
            score = cross_val_score(
                algorithm, dataset.X, dataset.y, n_jobs=-1, cv=train_test_splits, scoring=scoring
            )

            progress.update(loop_iterations, advance=1)
            return 1 - np.mean(score)

        return evaluate

    def replay_trajectory(self, trajectory: Trajectory, scoring, search_data, eval_data, progress, loop_iterations):

        scorer = get_scorer(scoring)
        results = []

        for item in trajectory.as_list:
            losses = []
            for seed in item.seeds:
                progress.update(loop_iterations, advance=1)
                algorithm = self.init_model(seed, search_data.metadata, **item.conf)
                algorithm.fit(search_data.X, search_data.y)
                loss = 1 - scorer(algorithm, eval_data.X, eval_data.y)
                losses.append(loss)

            results.append(dataclasses.replace(item, loss=np.mean(losses)))

        return Trajectory(results)
