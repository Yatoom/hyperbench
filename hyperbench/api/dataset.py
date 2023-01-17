from abc import abstractmethod, ABC
from dataclasses import dataclass
import numpy as np
import openml


@dataclass
class Data:
    X: np.ndarray
    y: np.ndarray
    categorical: list[int]
    numeric: list[int]
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
        self.task_id = task_id

    @property
    def name(self):
        return str(self.task_id)

    def load(self):
        task = openml.tasks.get_task(self.task_id)
        X, y = task.get_X_and_y()
        dataset = task.get_dataset()
        categorical = dataset.get_features_by_type("nominal", exclude=[task.target_name])
        numeric = dataset.get_features_by_type("numeric", exclude=[task.target_name])

        return Data(X, y, categorical, numeric, self)
