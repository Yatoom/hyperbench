from abc import ABC

from sklearn.linear_model import SGDClassifier
import ConfigSpace

from hyperbench.target_algorithms.base import BaseTarget


class SGD(BaseTarget):

    name = "SGD"
    deterministic = False

    @staticmethod
    def init_model(seed, metadata, **config):
        return SGDClassifier(**SGD.constants(), **config, random_state=seed)

    @staticmethod
    def constants():
        return {
            "loss": "log_loss",
            "max_iter": 1000,
            "learning_rate": "adaptive",
            "tol": None
        }

    @staticmethod
    def config_space():
        cs = ConfigSpace.ConfigurationSpace(seed=0)
        cs.add_hyperparameters([
            ConfigSpace.Float("alpha", (1e-5, 1), log=True, default=1e-3),
            ConfigSpace.Float("eta0", (1e-5, 1), log=True, default=1e-2)
        ])
        return cs
