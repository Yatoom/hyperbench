import dataclasses
from dataclasses import dataclass
from functools import cache

import numpy as np


@dataclass
class Dataset:
    id: str
    name: str
    X: np.ndarray
    y: np.ndarray
    categorical: list[int]
    numeric: list[int]
    provider: "Provider"

    def split(self, *lists_of_indices):
        for i in lists_of_indices:
            yield self.select(i)

    def select(self, indices):
        return dataclasses.replace(self, X=self.X[indices], y=self.y[indices])

    @property
    def n_rows(self) -> int:
        return self.X.shape[0]

    @property
    def n_columns(self) -> int:
        return self.X.shape[1]

    @property
    @cache
    def n_classes(self) -> int:
        return np.unique(self.y).shape[0]

    @property
    @cache
    def n_missing(self) -> int:
        return np.isnan(self.X).sum()
