import dataclasses
import json
from dataclasses import dataclass

import numpy as np


class Trajectory:

    def __init__(self, as_list):
        self.as_list = as_list

    def save(self, file: str):
        with open(file, "w+") as f:
            json.dump([dataclasses.asdict(e) for e in self.as_list], f, indent=2)
        return self

    @staticmethod
    def load(file: str):
        with open(file, "r") as f:
            data = json.load(f)
            entries = [Entry(e['conf'], e['loss'], e['at_iteration'], e['at_time'], e['seeds']) for e in data]
            return Trajectory(entries)

    def get_loss_over_time(self, max_time, step_size=1):

        x = np.arange(0, max_time, step_size)
        y = np.full_like(x, np.nan, dtype=float)

        for t in self.as_list[1:]:
            index = np.digitize(t.at_time, x)
            y[index:] = t.loss

        return x, y

    def get_loss_per_iteration(self, max_iter):

        x = np.arange(0, max_iter)
        y = np.zeros_like(x, dtype=float)

        for t in self.as_list:
            index = t.at_iteration
            y[index:] = t.loss

        return x[:-1], y[1:]


@dataclass
class Entry:
    conf: dict
    loss: float
    at_iteration: int
    at_time: any
    seeds: list[int]
