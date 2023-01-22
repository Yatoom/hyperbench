from abc import ABC, abstractmethod
from functools import cache

from hyperbench.dataset.dataset import Dataset


class Provider(ABC):

    @abstractmethod
    def __init__(self, dataset_id):
        self.id = dataset_id

    @property
    @abstractmethod
    def data(self) -> Dataset:
        pass

    @property
    @cache
    def stats(self) -> dict:
        return self.get_stats()

    @abstractmethod
    def get_stats(self):
        return {
            "id": self.id,
            "name": self.data.name,
            "n_rows": self.data.n_rows,
            "n_columns": self.data.n_columns,
            "n_classes": self.data.n_classes,
            "n_missing": self.data.n_missing,
            "n_numeric": len(self.data.numeric),
            "n_categorical": len(self.data.categorical)
        }
