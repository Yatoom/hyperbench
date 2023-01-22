from functools import cache

import openml
from hyperbench.dataset.dataset import Dataset
from hyperbench.provider.base import Provider


class OpenMLProvider(Provider):

    def __init__(self, dataset_id):
        self.task_id = dataset_id
        self.id = dataset_id
        self._data = None

    @property
    def stats(self):
        return {
            **self.default_stats(),
            "task_url": f"https://www.openml.org/t/{self.task_id}",
            "data_url": f"https://www.openml.org/d/{self._dataset.id}"
        }


    @property
    def data(self) -> Dataset:
        if self._data is None:
            X, y = self._xy
            self._data = Dataset(self.task_id, self._name, X, y, self._categorical, self._numeric, self)
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