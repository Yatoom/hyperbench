from .base import BaseTarget
from .rf import RandomForest
from .sgd import SGD
from .svm import SVM
from .xgboost import XGBoost


def get_target_by_name(name):
    if name == RandomForest.name:
        return RandomForest
    if name == SGD.name:
        return SGD
    if name == SVM.name:
        return SVM
    if name == XGBoost.name:
        return XGBoost
    return None