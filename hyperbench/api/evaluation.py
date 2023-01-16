from ConfigSpace import Configuration
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
import numpy as np

from hyperbench.api.trajectory import Trajectory


def get_config_evaluator(target_algorithm, data, benchmark):
    def evaluate(config: Configuration, seed: int):
        # Initialize algorithm
        algorithm = target_algorithm.initialize(seed, **dict(config))

        # Perform cross validation
        score = cross_val_score(
            algorithm, data.X, data.y, n_jobs=-1, cv=benchmark.train_test_splits, scoring=benchmark.scoring
        )

        print(score)

        return 1 - np.mean(score)

    return evaluate


def replay_trajectory(trajectory: Trajectory, scoring, target_algorithm, search_data, eval_data):

    scorer = get_scorer(scoring)
    results = []

    for item in trajectory.as_list:
        losses = []
        for seed in item['seeds']:
            algorithm = target_algorithm.initialize(seed, **dict(item['conf']))
            algorithm.fit(search_data.X, search_data.y)
            loss = 1 - scorer(algorithm, eval_data.X, eval_data.y)
            losses.append(loss)
        results.append({**item, "loss": np.mean(losses)})

    return Trajectory(results)
