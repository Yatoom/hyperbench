from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from smac.facade.smac_hpo_facade import SMAC4HPO

from hyperbench.api.benchmark import Benchmark, BenchmarkRunner
from hyperbench.api.dataset import OpenMLDataset
from hyperbench.api.optimizer import SMACBasedOptimizer
from hyperbench.api.target_algorithm import RandomForestFactory
from hyperbench.api.transformer import Imputer

tasks = [125920, 49, 146819, 29, 15, 3913, 3, 10101, 9971, 146818, 3917, 37, 3918, 14954, 9946, 146820, 3021, 31, 10093,
         3902, 3903, 9952, 9957, 167141, 14952, 9978, 3904, 43, 219, 14965, 7592]

benchmark = Benchmark(
    budget=10,
    transformer=Imputer(),
    scoring='balanced_accuracy',
    seeds=[2268061101, 2519249986, 338403738],
    target_algorithms=[RandomForestFactory.build()],
    datasets=[OpenMLDataset(task) for task in tasks],
    optimizers=[SMACBasedOptimizer(SMAC4HPO)],
    search_eval_splits=StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=0
    ),
    train_test_splits=ShuffleSplit(
        n_splits=3, random_state=0, test_size=0.10
    ),
)

BenchmarkRunner(benchmark).start()