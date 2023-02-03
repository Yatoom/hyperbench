import json
import os
import re
from datetime import timedelta
from functools import reduce

import numpy as np
import pandas as pd
import plotly.express as px

from hyperbench.dashboard.options import Options, Views
from hyperbench.trajectory.trajectory import Trajectory


def get_trajectories(path, iterations, time_based, details):
    entries = []

    details = details[1:]  # Skip target
    optimizer = details[0]
    multiplier = get_multiplier(optimizer)

    # Get trajectory for all multipliers
    for i in range(multiplier):
        speedup = i + 1
        if 1 < multiplier != speedup:
            details[0] = optimizer.split("_")[0] + f"_x{speedup}"
        else:
            details[0] = optimizer
        _, y = Trajectory.load(path).get_loss(iterations, time_based, speedup=speedup)

        virtual = speedup != multiplier
        entries.append([*details, virtual, y])

    return entries


def get_max_iterations(directory, filter_by_target):
    max_iter = 0
    max_time = 0
    for currentpath, folders, files in os.walk(directory):
        for file in files:

            if not file.endswith("search.json"):
                continue

            path = os.path.join(currentpath, file)
            details = path.replace(".json", "").split("/")[-5:]
            target, optimizer, seed, dataset, stage = details

            if not (target == filter_by_target):
                continue

            multiplier = get_multiplier(optimizer)
            max_iter = max(max_iter, Trajectory.load(path).max_iter / multiplier)
            max_time = max(max_time, Trajectory.load(path).max_time / multiplier)
    return max_iter, max_time


def load_trajectories(o: Options):
    directory, time_based, filter_by_target = o.directory, o.time_based, o.target
    trajectories = []

    max_iter, max_time = get_max_iterations(directory, o.target)
    maximum = max_time if time_based else max_iter

    for currentpath, folders, files in os.walk(directory):
        for file in files:

            if not (file.endswith("search.json") or file.endswith("eval.json")):
                continue

            path = os.path.join(currentpath, file)
            details = path.replace(".json", "").split("/")[-5:]
            target, optimizer, seed, dataset, stage = details

            if not (target == filter_by_target):
                continue

            for trajectory in get_trajectories(path, maximum, time_based, details):
                trajectories.append(trajectory)

    df = pd.DataFrame(trajectories, columns=["optimizer", "seed", "dataset", "stage", "virtual", "trajectory"])
    expanded = pd.concat([df.drop('trajectory', axis=1), df['trajectory'].apply(pd.Series)], axis=1)

    if o.view == Views.LIVE:
        expanded = live_view(expanded)
    elif o.view == Views.STATIC:
        expanded = static_view(expanded)

    return expanded


def load_stats(directory, dataframe, target):
    indices = dataframe[["optimizer", "seed", "dataset", "virtual"]].copy()
    indices = indices[~indices.virtual]
    indices.loc[:, "target"] = target
    rows = []
    for index in indices.iloc:
        path = os.path.join(directory, index.target, index.optimizer, index.seed, index.dataset, "stats.json")
        with open(path, "r") as f:
            row = json.load(f)
            row = {**index, **row}
            rows.append(row)
    return pd.DataFrame(rows)


def format_stats_table(df, time_cols=None, int_cols=None):
    res = df.copy()
    if time_cols is None:
        time_cols = ["ta_time_used", "wallclock_time_used", "perf_time"]
        time_cols = [i for i in time_cols if i in df.columns]
    if int_cols is None:
        int_cols = ["submitted_ta_runs", "finished_ta_runs"]
        int_cols = [i for i in int_cols if i in df.columns]
    res.loc[:, time_cols] = res[time_cols].applymap(lambda x: str(timedelta(seconds=np.round(x))))
    res.loc[:, int_cols] = res[int_cols].astype('int')
    return res


def get_dataset_stats(df):
    stats = df.groupby(["optimizer", "dataset"])[
        ["ta_time_used", "wallclock_time_used", "submitted_ta_runs"]].mean(numeric_only=False) \
        .sort_values(by=["optimizer", "wallclock_time_used"], ascending=False).reset_index().set_index("optimizer")
    return format_stats_table(stats)


def get_run_stats(df):
    stats = df.groupby("optimizer")[
        ["submitted_ta_runs", "finished_ta_runs", "ta_time_used", "wallclock_time_used"]] \
        .agg(['mean', 'std'])
    return format_stats_table(stats)


def get_other_stats(df):
    return df.groupby("optimizer")[["mean_cost", "inc_changed", "n_configs"]] \
        .agg(lambda x: f"{np.mean(x):.2f} ± {np.std(x):.2f}")


def filter_on(dataframe, **kwargs):
    res = dataframe.copy()
    for k, v in kwargs.items():
        res = res[res[k] == v].drop(k, axis=1)

    return res


def live_view(dataframe):
    datasets_per_group = list(dataframe.groupby(["optimizer", "seed", "stage"])["dataset"].agg(list)
                              .reset_index(drop=True).to_dict().values())
    common_datasets = reduce(np.intersect1d, datasets_per_group)
    return dataframe[dataframe["dataset"].isin(common_datasets)]


def static_view(dataframe):
    indexed = dataframe.set_index(["optimizer", "seed"])
    counts = filter_on(indexed, stage="eval").groupby(["optimizer", "seed"]).dataset.count()
    filtered = indexed[counts == counts.max()]
    return filtered.reset_index()


def get_target_algorithms(directory):
    return [f for f in os.listdir(directory) if not f.startswith(".")]


def get_datasets(dataframe):
    return dataframe.dataset.unique()


def overview(dataframe):
    filtered = filter_on(dataframe, stage="eval")
    return filtered.set_index(["optimizer", "seed"])["dataset"] \
        .groupby(["optimizer", "seed"]).count().reset_index().rename({"dataset": "datasets"}, axis=1)


def normalize(dataframe):
    return dataframe.set_index(["optimizer", "dataset", "seed"]).groupby(["dataset"]) \
        .transform(lambda x: (x - x.mean()) / x.std()).reset_index()


def rank(dataframe):
    return dataframe.set_index(['optimizer', 'seed', 'dataset']).groupby(['dataset']).rank(axis=0) \
        .reset_index()


def aggregate_over_seeds(dataframe):
    # Needs dataframe with columns optimizer, dataset, seed, *trajectory
    return dataframe.groupby(['optimizer', 'dataset']).mean(numeric_only=True).reset_index()


def aggregate_over_datasets(dataframe):
    # Needs dataframe with columns optimizer, dataset, *trajectory
    return dataframe.groupby(['optimizer']).mean(numeric_only=True).reset_index()


def visualize(dataframe):
    frame = dataframe.set_index("optimizer").T
    fig = px.line(frame)
    return fig


def get_multiplier(optimizer: str):
    """
    If the optimizer has e.g. _x2 or _x3 in its name, this means that the optimizer has been given a multiple of the
    budget of the others in the experiment. This function will retrieve that multiplier from its name.

    Parameters
    ----------
    optimizer: str
        The name of the optimizer

    Returns
    -------
    multiplier: int
        The multiplier retrieved from the name
    """
    has_multiplied_budget = bool(re.match(".*_x\d*", optimizer))
    multiplier = 1 if not has_multiplied_budget else int(re.search("_x\d*", optimizer)[0].replace("_x", ""))
    return multiplier
