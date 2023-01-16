from abc import abstractmethod
from dataclasses import dataclass

from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TargetAlgorithm:
    name: str
    model: any
    preset_params: dict
    config_space: ConfigurationSpace
    is_deterministic: bool

    def initialize(self, seed, **config):
        return self.model(random_state=seed, **self.preset_params, **config)


class TargetAlgorithmFactory:
    @staticmethod
    @abstractmethod
    def build() -> TargetAlgorithm:
        pass


class RandomForestFactory(TargetAlgorithmFactory):
    @staticmethod
    def build():
        return TargetAlgorithm(
            name="RandomForest",
            model=RandomForestClassifier,
            config_space=RandomForestFactory.config_space(),
            preset_params=RandomForestFactory.constants(),
            is_deterministic=False,
        )

    @staticmethod
    def constants():
        return {
            "max_depth": None,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "min_weight_fraction_leaf": 0.0,
            "n_estimators": 100,
            "verbose": 0,
            "n_jobs": -1,
        }

    @staticmethod
    def config_space():
        config_space = ConfigurationSpace()

        criterion = Categorical("criterion", ["gini", "entropy"], default="gini")
        max_features = Float("max_features", (0.0, 1.0), default=0.5)
        min_samples_split = Integer("min_samples_split", (2, 20), default=2)
        min_samples_leaf = Integer("min_samples_leaf", (1, 20), default=1)
        bootstrap = Categorical("bootstrap", [True, False], default=True)

        config_space.add_hyperparameters(
            [criterion, max_features, min_samples_split, min_samples_leaf, bootstrap]
        )
        return config_space
