from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cache

import numpy as np
import openml


@dataclass
class Data:
    X: np.ndarray
    y: np.ndarray
    categorical: np.array
    numeric: np.array
    parent: "Dataset"

    def split(self, *lists_of_indices):
        for i in lists_of_indices:
            yield self.select(i)

    def select(self, indices):
        return Data(self.X[indices], self.y[indices], self.categorical, self.numeric, self.parent)


class Dataset(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def data(self) -> Data:
        pass


class OpenMLDataset(Dataset, ABC):

    def __init__(self, task_id):
        self.task_id = task_id

    @property
    def id(self):
        return self.task_id

    @property
    @cache
    def task(self):
        return openml.tasks.get_task(self.id)

    @property
    @cache
    def dataset(self):
        return self.task.get_dataset()

    @property
    @cache
    def name(self):
        return self.dataset.name

    @property
    @cache
    def data(self):
        X, y = self.task.get_X_and_y()
        categorical = self.dataset.get_features_by_type("nominal", exclude=[self.task.target_name])
        numeric = self.dataset.get_features_by_type("numeric", exclude=[self.task.target_name])
        return Data(X, y, categorical, numeric, self)

    @property
    @cache
    def n_rows(self):
        return self.data.X.shape[0]

    @property
    @cache
    def n_columns(self):
        return self.data.X.shape[1]

    @property
    @cache
    def n_classes(self):
        return np.unique(self.data.y).shape[0]

    @property
    @cache
    def n_missing(self):
        return np.isnan(self.data.X).sum()

    @property
    @cache
    def stats(self):
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_classes": self.n_classes,
            "n_missing": self.n_missing,
            "n_numeric": len(self.data.numeric),
            "n_categorical": len(self.data.categorical),
            "task_url": f"https://www.openml.org/t/{self.id}",
            "data_url": f"https://www.openml.org/d/{self.dataset.id}",
        }
