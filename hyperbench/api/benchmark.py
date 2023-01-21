import dataclasses
import json
import os
from dataclasses import dataclass

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn, TimeElapsedColumn
from sklearn.model_selection import BaseShuffleSplit

from hyperbench.api.dataset import Dataset
from hyperbench.api.evaluation import get_config_evaluator, replay_trajectory
from hyperbench.api.optimizer import Optimizer
from hyperbench.api.target_algorithm import TargetAlgorithm
from hyperbench.api.transformer import Transformer
import numpy as np


@dataclass
class Benchmark:
    budget: int
    transformer: Transformer
    scoring: str
    output_folder: str

    seeds: list[int]
    target_algorithms: list[TargetAlgorithm]
    datasets: list[Dataset]
    optimizers: list[Optimizer]
    search_eval_splits: BaseShuffleSplit
    train_test_splits: BaseShuffleSplit


class BenchmarkRunner:

    def __init__(self, benchmark):
        self.benchmark = benchmark
        progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(),
                            TimeRemainingColumn(), TimeElapsedColumn())

        self.track_seeds = progress.add_task("Seeds", total=len(self.benchmark.seeds))
        self.track_targets = progress.add_task("Targets", total=len(self.benchmark.target_algorithms))
        self.track_data = progress.add_task("Datasets", total=len(self.benchmark.datasets))
        self.track_opt = progress.add_task("Optimizers", total=len(self.benchmark.optimizers))
        self.track_splits = progress.add_task("Splits", total=self.benchmark.search_eval_splits.n_splits)
        self.track_stage = progress.add_task("Stage", total=2)
        self.track_iterations = progress.add_task("Evaluations", total=self.benchmark.budget)
        self.progress = progress

    def start(self):
        with self.progress as progress:
            self.loop_seeds()

    def loop_seeds(self):
        self.progress.reset(self.track_seeds)
        for seed in self.benchmark.seeds:
            self.loop_targets(seed)
            self.progress.update(self.track_seeds, advance=1)

    def loop_targets(self, seed):
        self.progress.reset(self.track_targets)
        for target in self.benchmark.target_algorithms:
            self.loop_datasets(seed, target)
            self.progress.update(self.track_targets, advance=1)

    def loop_datasets(self, seed, target):
        self.progress.reset(self.track_data)
        for dataset in self.benchmark.datasets:
            data = dataset.load()
            self.loop_optimizers(seed, target, data)
            self.progress.update(self.track_data, advance=1)

    def loop_optimizers(self, seed, target, dataset):
        self.progress.reset(self.track_opt)
        for optimizer in self.benchmark.optimizers:
            self.loop_splits(seed, target, dataset, optimizer)
            self.progress.update(self.track_opt, advance=1)

    def loop_splits(self, seed, target, dataset, optimizer):
        self.progress.reset(self.track_splits)
        for search_indices, eval_indices in self.benchmark.search_eval_splits.split(dataset.X, dataset.y):
            search_set, eval_set = dataset.split(search_indices, eval_indices)
            new_search_set, new_eval_set = self.benchmark.transformer.transform(search_set, eval_set)

            self.progress.reset(self.track_stage)
            search_trajectory, stats = self.search_stage(seed, target, new_search_set, optimizer)
            self.progress.update(self.track_stage, advance=1)
            eval_trajectory = self.evaluation_stage(target, new_search_set, new_eval_set, search_trajectory)
            self.progress.update(self.track_stage, advance=1)

            self.save(search_trajectory, seed, target.name, dataset.parent.name, optimizer.name, "search")
            self.save(eval_trajectory, seed, target.name, dataset.parent.name, optimizer.name, "eval")
            self.save_stats(stats, seed, target.name, dataset.parent.name, optimizer.name)
            self.progress.update(self.track_splits, advance=1)

    def save(self, trajectory, seed: int, target: str, dataset: str, optimizer: str, stage: str):
        seed = str(seed)
        path = os.path.join(self.benchmark.output_folder, target, optimizer, seed, dataset)
        file = os.path.join(path, f"{stage}.json")
        os.makedirs(path, exist_ok=True)
        trajectory.save(file)

    def save_stats(self, stats, seed, target, dataset, optimizer):
        seed = str(seed)
        path = os.path.join(self.benchmark.output_folder, target, optimizer, seed, dataset)
        file = os.path.join(path, "stats.json")
        os.makedirs(path, exist_ok=True)
        with open(file, "w+") as f:
            json.dump(stats, f, indent=2)

    def search_stage(self, seed, target, dataset, optimizer):
        tae_runner = get_config_evaluator(target, dataset, self.benchmark, self.progress, self.track_iterations)
        optimizer.initialize(tae_runner, seed, dataset, self.benchmark.budget, target)
        optimizer.search()
        self.progress.reset(self.track_iterations)
        return optimizer.get_trajectory(), optimizer.get_stats()

    def evaluation_stage(self, target, search_set, eval_set, search_trajectory):
        eval_trajectory = replay_trajectory(search_trajectory, self.benchmark.scoring, target, search_set, eval_set)
        return eval_trajectory
