import numpy as np

import openml
from hyperbench.dataset.dataset import Dataset
from hyperbench.dataset.metadata import Metadata
from hyperbench.provider.base import Provider


class OpenMLProvider(Provider):

    def __init__(self, dataset_id):
        self.task_id = dataset_id
        self.id = dataset_id
        self._data = None
        self._metadata = None

    @property
    def stats(self):
        return {
            **self.default_stats(),
            "task_url": f"https://www.openml.org/t/{self.id}",
            "data_url": f"https://www.openml.org/d/{self._dataset.id}"
        }

    @property
    def data(self) -> Dataset:
        if self._data is None:
            X, y = self._xy
            self._metadata = self._get_metadata(X, y)
            self._data = Dataset(X, y, self._metadata)
        return self._data

    @property
    def _task(self):
        return openml.tasks.get_task(self.task_id)

    @property
    def _dataset(self):
        return self._task.get_dataset()

    @property
    def _xy(self):
        return self._task.get_X_and_y()

    @property
    def _numeric(self):
        return self._dataset.get_features_by_type("numeric", exclude=[self._task.target_name])

    @property
    def _categorical(self):
        return self._dataset.get_features_by_type("nominal", exclude=[self._task.target_name])

    @property
    def _name(self):
        return self._dataset.name

    def _get_metadata(self, X, y):
        return Metadata(
            id=self.task_id,
            name=self._name,
            categorical=self._categorical,
            numeric=self._numeric,
            n_rows=X.shape[0],
            n_columns=X.shape[1],
            n_classes=np.unique(y).shape[0],
            n_missing=np.isnan(X).sum()
        )
