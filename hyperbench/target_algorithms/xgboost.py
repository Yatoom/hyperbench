import ConfigSpace
from sklearn.svm import SVC

from hyperbench.target_algorithms.base import BaseTarget
from xgboost import XGBClassifier


class XGBoost(BaseTarget):

    name = "XGBoost"
    deterministic = False

    @staticmethod
    def init_model(seed, metadata, **config):

        constants = XGBoost.constants()

        if metadata.n_classes > 2:
            constants["objective"] = "multi:softmax"
            constants["num_class"] = metadata.n_classes

        return XGBClassifier(**constants, **config, random_state=seed)

    @staticmethod
    def constants():
        return {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "n_estimators": 2000,
            "subsample": 1
        }

    @staticmethod
    def config_space():
        cs = ConfigSpace.ConfigurationSpace(seed=0)

        cs.add_hyperparameters([
            ConfigSpace.Float('eta', (2 ** -10, 1.), default=-.3, log=True),  # learning rate
            ConfigSpace.Integer('max_depth', (1, 50), default=10, log=True),
            ConfigSpace.Float('colsample_bytree', (0.1, 1.), default=1., log=False),
            ConfigSpace.Float('reg_lambda', (2 ** -10, 2 ** 10), default=1, log=True)
        ])

        return cs
