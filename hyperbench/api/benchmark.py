from dataclasses import dataclass
from sklearn.model_selection import BaseShuffleSplit

from hyperbench.api.dataset import Dataset
from hyperbench.api.optimizer import Optimizer
from hyperbench.api.target_algorithm import TargetAlgorithm
from hyperbench.api.transformer import Transformer


@dataclass
class Benchmark:
    budget: int
    transformer: Transformer
    scoring: str

    seeds: list[int]
    target_algorithms: list[TargetAlgorithm]
    datasets: list[Dataset]
    optimizers: list[Optimizer]
    search_eval_splits: BaseShuffleSplit
    train_test_splits: BaseShuffleSplit

