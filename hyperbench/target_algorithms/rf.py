from sklearn.linear_model import SGDClassifier
import ConfigSpace

from hyperbench.target_algorithms.base import BaseTarget


class RandomForest(BaseTarget):

    name = "RandomForest"
    deterministic = False

    @staticmethod
    def init_model(seed, **config):

        # Find out what to do with this
        # n_features = self.train_X.shape[1]
        # config["max_features"] = int(np.rint(np.power(n_features, config["max_features"])))

        return SGDClassifier(**RandomForest.constants(), **config, random_state=seed)

    @staticmethod
    def constants():
        return {
            "n_estimators": 100,
            "bootstrap": True
        }

    @staticmethod
    def config_space():
        cs = ConfigSpace.ConfigurationSpace(seed=0)
        cs.add_hyperparameters([
            ConfigSpace.Integer('max_depth', (1, 50), default=10, log=True),
            ConfigSpace.Integer('min_samples_split', (2, 128), default=32, log=True),
            ConfigSpace.Float('max_features', (0, 1.0), default=0.5, log=False),
            ConfigSpace.Integer('min_samples_leaf', (1, 20), default=1, log=False),
        ])
        return cs

