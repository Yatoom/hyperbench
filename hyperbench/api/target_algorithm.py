from abc import abstractmethod
from dataclasses import dataclass

from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, EqualsCondition
from catboost import CatBoostClassifier
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

class CatboostFactory(TargetAlgorithmFactory):
    @staticmethod
    def build():
        return TargetAlgorithm(
            name="Catboost",
            model=CatBoostClassifier,
            config_space=CatboostFactory.config_space(),
            preset_params=CatboostFactory.constants(),
            is_deterministic=False,
        )
    @staticmethod
    def constants():
        return {
            "used_ram_limit": "3gb",
            "eval_metric": "Accuracy",
            "num_trees": 100,
            "verbose": False,
        }

    @staticmethod
    def config_space():
        config_space = ConfigurationSpace()

        objective = Categorical("objective", ["MultiClass", "CrossEntropy"])
        colsample_bylevel = Float('colsample_bylevel', bounds=(0.01, 0.1), log=True)
        depth = Integer('depth', bounds=(1, 12))
        boosting_type = Categorical('boosting_type', ["Ordered", "Plain"])
        bootstrap_type = Categorical('bootstrap_type', ["Bayesian", "Bernoulli", "MVS"])
        bagging_temperature = Float('bagging_temperature', bounds=(0, 10))
        subsample = Float("subsample", bounds=(0.1, 1), log=True)
        learning_rate = Float("learning_rate", bounds=(0.001, 1), log=True)

        c1 = EqualsCondition(bagging_temperature, bootstrap_type, "Bayesian")
        c2 = EqualsCondition(subsample, bootstrap_type, "Bernoulli")

        config_space.add_hyperparameters([
            objective, colsample_bylevel, depth, learning_rate,
            boosting_type, bootstrap_type, bagging_temperature, subsample
        ])

        config_space.add_conditions([c1, c2])
        return config_space