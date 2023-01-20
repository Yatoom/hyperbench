import os
from functools import reduce

import numpy as np
import pandas as pd
import plotly.express as px
from hyperbench.api.trajectory import Trajectory


def get_all_trajectories(directory, iterations=300):
    entries = []
    for currentpath, folders, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(currentpath, file)
                details = path.replace(".json", "").split("/")[-5:]
                _, y = Trajectory.load(path).get_loss_per_iteration(iterations)
                entries.append([*details, y])
    df = pd.DataFrame(entries, columns=["target", "optimizer", "seed", "dataset", "stage", "trajectory"])
    expanded = pd.concat([df.drop('trajectory', axis=1), df['trajectory'].apply(pd.Series)], axis=1)
    return expanded


def filter_on(dataframe, **kwargs):
    res = dataframe.copy()
    for k, v in kwargs.items():
        res = res[res[k] == v].drop(k, axis=1)

    return res


def intersect(dataframe):
    datasets_per_group = list(dataframe.groupby(["optimizer", "seed", "stage"])["dataset"].agg(list)
                              .reset_index(drop=True).to_dict().values())
    common_datasets = reduce(np.intersect1d, datasets_per_group)
    return dataframe[dataframe["dataset"].isin(common_datasets)]


def union(dataframe):
    stats = overview(dataframe)
    filtered = stats[stats["datasets"] == stats["datasets"].max()]
    tags = filtered.optimizer + filtered.seed
    return dataframe[(dataframe.optimizer + dataframe.seed).isin(tags)]


def get_target_algorithms(dataframe):
    return dataframe.target.unique()


def overview(dataframe):
    filtered = filter_on(dataframe, stage="eval")
    return filtered.set_index(["optimizer", "seed"])["dataset"] \
        .groupby(["optimizer", "seed"]).count().reset_index().rename({"dataset": "datasets"}, axis=1)


def normalize(dataframe):
    return dataframe.set_index(["optimizer", "dataset", "seed"]).groupby(["dataset"]) \
        .transform(lambda x: (x - x.mean()) / x.std()).reset_index()


def rank(dataframe):
    return dataframe.set_index(['optimizer', 'seed', 'dataset']).groupby(['dataset']).rank() \
        .reset_index()


def aggregate_over_seeds(dataframe):
    # Needs dataframe with columns optimizer, dataset, seed, *trajectory
    return dataframe.groupby(['optimizer', 'dataset']).mean().reset_index()


def aggregate_over_datasets(dataframe):
    # Needs dataframe with columns optimizer, dataset, *trajectory
    return dataframe.groupby(['optimizer']).mean().reset_index()


def visualize(dataframe):
    frame = dataframe.set_index("optimizer").T
    fig = px.line(frame)
    return fig
