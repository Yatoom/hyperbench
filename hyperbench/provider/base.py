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
    @abstractmethod
    def stats(self) -> dict:
        return self.default_stats()

    def default_stats(self):
        metadata = self.data.metadata
        return {
            "id": metadata.id,
            "name": metadata.name,
            "n_rows": metadata.n_rows,
            "n_columns": metadata.n_columns,
            "n_classes": metadata.n_classes,
            "n_missing": metadata.n_missing,
            "n_numeric": len(metadata.numeric),
            "n_categorical": len(metadata.categorical)
        }
