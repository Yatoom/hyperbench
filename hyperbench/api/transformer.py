import dataclasses
from abc import abstractmethod, ABC

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from hyperbench.api.dataset import Data


class Transformer(ABC):

    def transform(self, search_set: Data, eval_set: Data):
        search_set_copy = dataclasses.replace(search_set)
        eval_set_copy = dataclasses.replace(search_set)

        return self.update(search_set_copy, eval_set_copy)

    @abstractmethod
    def update(self, search_set: Data, eval_set: Data):
        pass


class Imputer(Transformer):

    def update(self, search_set: Data, eval_set: Data):
        ct = ColumnTransformer([
            ["most_frequent", SimpleImputer(strategy="most_frequent"), search_set.categorical],
            ["median", SimpleImputer(strategy="median"), eval_set.numeric]
        ])

        search_set.X = ct.fit_transform(search_set.X)
        eval_set.X = ct.transform(eval_set.X)

        return search_set, eval_set
