import dataclasses
from abc import ABC, abstractmethod

from hyperbench.dataset.dataset import Dataset


class Transformer(ABC):

    def transform(self, search_set: Dataset, eval_set: Dataset):
        search_set_copy = dataclasses.replace(search_set)
        eval_set_copy = dataclasses.replace(eval_set)
        return self.update(search_set_copy, eval_set_copy)

    @abstractmethod
    def update(self, search_set: Dataset, eval_set: Dataset):
        pass
