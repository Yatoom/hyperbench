from abc import abstractmethod, ABC
from dataclasses import dataclass
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

    @abstractmethod
    def load(self) -> Data:
        pass


class OpenMLDataset(Dataset):

    def __init__(self, task_id):
        self.id = task_id
        self._name = None

    @property
    def name(self):
        return self._name

    def load(self):
        task = openml.tasks.get_task(self.id)
        dataset = task.get_dataset()
        self._name = dataset.name
        X, y = task.get_X_and_y()
        categorical = dataset.get_features_by_type("nominal", exclude=[task.target_name])
        numeric = dataset.get_features_by_type("numeric", exclude=[task.target_name])
        return Data(X, y, categorical, numeric, self)
