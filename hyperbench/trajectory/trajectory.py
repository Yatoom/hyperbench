import dataclasses
import json

import numpy as np

from hyperbench.trajectory.entry import Entry


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

    def get_loss(self, max_budget, time_based=False, step_size=1, speedup=1):
        if time_based:
            return self.get_loss_over_time(max_budget, step_size, speedup=speedup)
        return self.get_loss_per_iteration(max_budget, speedup=speedup)

    def get_loss_over_time(self, max_time, step_size=1, speedup=1):

        x = np.arange(0, max_time, step_size)
        y = np.full_like(x, np.nan, dtype=float)

        for t in self.as_list:
            if t.at_time / speedup > max_time:
                return x, y
            index = np.digitize(t.at_time / speedup, x)
            y[index:] = t.loss

        return x, y

    def get_loss_per_iteration(self, max_iter, speedup=1):

        x = np.arange(0, max_iter)
        y = np.full_like(x, np.nan, dtype=float)

        for t in self.as_list:
            if t.at_iteration / speedup > max_iter:
                return x, y
            index = np.digitize(t.at_iteration / speedup, x, right=True)
            y[index:] = t.loss

        return x, y
