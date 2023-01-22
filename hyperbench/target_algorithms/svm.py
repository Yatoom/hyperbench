import ConfigSpace
from sklearn.svm import SVC

from hyperbench.target_algorithms.base import BaseTarget


class SVM(BaseTarget):

    name = "SVC"
    deterministic = False

    @staticmethod
    def init_model(seed, **config):
        return SVC(**SVM.constants(), **config, random_state=seed)

    @staticmethod
    def constants():
        return {
            "cache_size": 200
        }

    @staticmethod
    def config_space():
        cs = ConfigSpace.ConfigurationSpace(seed=0)
        cs.add_hyperparameters([
            ConfigSpace.Float("C", (2 ** -10, 2 ** 10), log=True, default=1.0),
            ConfigSpace.Float("gamma", (2 ** -10, 2 ** 10), log=True, default=0.1)
        ])
        return cs
