from dataclasses import dataclass

from sklearn.model_selection import BaseShuffleSplit

from hyperbench.optimizers.base import Optimizer
from hyperbench.provider import Provider
from hyperbench.target_algorithms import BaseTarget
from hyperbench.transformer.base import Transformer


@dataclass
class BenchmarkConfig:
    budget: int
    time_based: bool
    transformer: Transformer
    scoring: str
    output_folder: str

    seeds: list[int]
    target_algorithms: list[BaseTarget]
    datasets: list[Provider]
    optimizers: list[Optimizer]
    search_eval_splits: BaseShuffleSplit
    train_test_splits: BaseShuffleSplit

