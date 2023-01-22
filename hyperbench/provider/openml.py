from functools import cache

import openml
from hyperbench.dataset.dataset import Dataset
from hyperbench.provider.base import Provider


class OpenMLProvider(Provider):

    def __init__(self, dataset_id):
        self.task_id = dataset_id

    def get_stats(self):
        return {
            **self.stats,
            "task_url": f"https://www.openml.org/t/{self.id}",
            "data_url": f"https://www.openml.org/d/{self._dataset.id}"
        }

    @property
    @cache
    def data(self) -> Dataset:
        X, y = self._xy
        return Dataset(self.task_id, self._name, X, y, self._categorical, self._numeric, self)

    @property
    @cache
    def _task(self):
        return openml.tasks.get_task(self.task_id)

    @property
    @cache
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
    @cache
    def _name(self):
        return self._dataset.name