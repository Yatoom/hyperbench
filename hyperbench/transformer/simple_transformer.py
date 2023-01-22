from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from hyperbench.dataset.dataset import Dataset
from hyperbench.transformer.base import Transformer


class SimpleTransformer(Transformer):

    def update(self, search_set: Dataset, eval_set: Dataset):
        ct = ColumnTransformer([
            ["most_frequent", SimpleImputer(strategy="most_frequent"), search_set.metadata.categorical],
            ["median", SimpleImputer(strategy="median"), eval_set.metadata.numeric]
        ])

        search_set.X = ct.fit_transform(search_set.X)
        eval_set.X = ct.transform(eval_set.X)

        return search_set, eval_set
