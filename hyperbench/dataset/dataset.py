import dataclasses
from dataclasses import dataclass
import numpy as np

from hyperbench.dataset.metadata import Metadata


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    metadata: Metadata

    def split(self, *lists_of_indices):
        for i in lists_of_indices:
            yield self.select(i)

    def select(self, indices):
        return dataclasses.replace(self, X=self.X[indices], y=self.y[indices])
