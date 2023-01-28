import os
from datetime import timedelta
import numpy as np

import pandas as pd
from smac.runhistory.runhistory import RunHistory
import ConfigSpace

from hyperbench.target_algorithms import SGD, BaseTarget, get_target_by_name


def load_leaderboard(folder: str, target: str, optimizer: str, seed: str, dataset: str):
    cs = get_target_by_name(target).config_space()
    path = os.path.join(folder, target, optimizer, seed, dataset)
    path = os.path.join(path, os.listdir(path)[0], "runhistory.json")
    history = RunHistory()
    history.load_json(path, cs)

    history = [
        {
            "config_id": key.config_id,
            "config": history.ids_config[key.config_id]._values,
            "seed": key.seed,
            "budget": key.budget,
            "cost": value.cost,
            "time": value.time,
            "status": value.status
        }

        for key, value in history.data.items()
    ]
    frame = pd.DataFrame(history)
    frame = pd.concat([frame, frame.config.apply(pd.Series)], axis=1)
    frame = frame.drop(columns=["config"])
    frame["time"] = frame.time.apply(lambda x: str(timedelta(seconds=np.round(x))))
    return frame


# frame = load_leaderboard("../../smac_output", "SGD", "roar_x2", "2268061101", "adult")


# print()