from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from smac.facade.smac_hpo_facade import SMAC4HPO

from hyperbench.benchmark import BenchmarkRunner, BenchmarkConfig
from hyperbench.hyperboost import HyperboostEPM
from hyperbench.optimizers import SMACBasedOptimizer
from hyperbench.provider import OpenMLProvider
from hyperbench.target_algorithms import SVM
from hyperbench.transformer import SimpleTransformer

# This error was related to subsample rate nog being high enough.


benchmark = BenchmarkConfig(
    budget=300,
    time_based=False,
    transformer=SimpleTransformer(),
    output_folder="results-test",
    scoring="balanced_accuracy",

    seeds=[2268061101],
    target_algorithms=[SVM()],
    datasets=[OpenMLProvider(16)],  # mfeat-karhunen
    optimizers=[
        SMACBasedOptimizer(SMAC4HPO, "hyperboost", budget_multiplier=1, model=HyperboostEPM)
    ],
    search_eval_splits=StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=0
    ),
    train_test_splits=ShuffleSplit(
        n_splits=3, random_state=0, test_size=0.10
    ),
)


BenchmarkRunner(benchmark).start()
