import os
import pandas as pd
import plotly.express as px
from hyperbench.api.trajectory import Trajectory


def get_all_trajectories(directory):
    entries = []
    for currentpath, folders, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(currentpath, file)
                details = path.replace(".json", "").split("/")[-5:]
                _, y = Trajectory.load(path).get_loss_per_iteration(10)
                entries.append([*details, y])
    df = pd.DataFrame(entries, columns=["optimizer", "target", "seed", "dataset", "stage", "trajectory"])
    expanded = pd.concat([df.drop('trajectory', axis=1), df['trajectory'].apply(pd.Series)], axis=1)
    return expanded


def filter_on(dataframe, **kwargs):
    res = dataframe.copy()
    for k, v in kwargs.items():
        res = res[res[k] == v].drop(k, axis=1)

    return res


def aggregate_over_seeds(dataframe):
    # Needs dataframe with columns optimizer, target, dataset, seed, *trajectory
    return dataframe.groupby(['optimizer', 'target', 'dataset']).mean().reset_index()


def aggregate_over_datasets(dataframe):
    # Needs dataframe with columns optimizer, target, dataset, *trajectory
    return dataframe.groupby(['optimizer', 'target']).mean().reset_index()


def visualize(dataframe, target):
    frame = filter_on(dataframe, target=target).set_index("optimizer").T
    fig = px.line(frame)
    return fig
