from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical, EqualsCondition
from catboost import CatBoostClassifier

from hyperbench.target_algorithms.target_algorithm import TargetAlgorithm
from hyperbench.target_algorithms.target_algorithm_factory import TargetAlgorithmFactory


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
            "objective": "MultiClass"
        }

    @staticmethod
    def config_space():
        config_space = ConfigurationSpace()

        colsample_bylevel = Float('colsample_bylevel', bounds=(0.01, 0.1), log=True)
        depth = Integer('depth', bounds=(4, 10))
        boosting_type = Categorical('boosting_type', ["Ordered", "Plain"])
        bootstrap_type = Categorical('bootstrap_type', ["Bayesian", "Bernoulli", "MVS"])
        bagging_temperature = Float('bagging_temperature', bounds=(0, 10))
        subsample = Float("subsample", bounds=(0.1, 1), log=True)
        learning_rate = Float("learning_rate", bounds=(0.01, 1), log=True)
        random_strength = Integer("random_strength", bounds=(0, 1))

        c1 = EqualsCondition(bagging_temperature, bootstrap_type, "Bayesian")
        c2 = EqualsCondition(subsample, bootstrap_type, "Bernoulli")

        config_space.add_hyperparameters([
            colsample_bylevel, depth, learning_rate,
            boosting_type, bootstrap_type, bagging_temperature, subsample, random_strength
        ])

        config_space.add_conditions([c1, c2])
        return config_space
