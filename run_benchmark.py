import openml
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from smac.facade.roar_facade import ROAR
from smac.facade.smac_hpo_facade import SMAC4HPO

from hyperbench.benchmark import BenchmarkConfig
from hyperbench.benchmark import BenchmarkRunner
from hyperbench.hyperboost import HyperboostEPM
from hyperbench.provider import OpenMLProvider
from hyperbench.optimizers import SMACBasedOptimizer
from hyperbench.target_algorithms import SVM, RandomForest, SGD, XGBoost
from hyperbench.transformer import SimpleTransformer

tasks = openml.study.get_suite(99).tasks

# These datasets take too long to evaluate on
excluded = [
    # Dimensions are too high
    167124,	 # CIFAR_10
    167121,	 # Devnagari-Script
    146825,	 # Fashion-MNIST
    3573,	 # mnist_784
    9910,	 # Bioresponse
    14970,	 # har
    167125,	 # Internet-Advertisements
    3481,	 # isolet
    9977,	 # nomao
    146195,	 # connect-4
    167120,	 # numerai28.6
    9976,	 # madelon
    9981,	 # cnae-9
    6,	     # letter

    # Too many rows for SVM classifier
    7592,    # adult
    219,     # electricity
    14965,   # bank-marketing
    167119,  # jungle_chess_2pcs_raw_endgame_complete
]
tasks = [task for task in tasks if task not in excluded]


benchmark = BenchmarkConfig(
    budget=300,
    time_based=False,
    transformer=SimpleTransformer(),
    output_folder="results",
    scoring="balanced_accuracy",  # make_scorer(roc_auc_score, multi_class='ovo'

    seeds=[2268061101, 2519249986, 338403738],
    target_algorithms=[RandomForest(), XGBoost(), SGD(), SVM()],
    datasets=[OpenMLProvider(task) for task in tasks],
    optimizers=[
        # SMACBasedOptimizer(ROAR, "roar_x1", budget_multiplier=1)
        SMACBasedOptimizer(ROAR, "roar_x2", budget_multiplier=2),
        SMACBasedOptimizer(SMAC4HPO, "smac", budget_multiplier=1),
        SMACBasedOptimizer(SMAC4HPO, "hyperboost", budget_multiplier=1, model=HyperboostEPM)
    ],
    search_eval_splits=StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=0
    ),
    train_test_splits=ShuffleSplit(
        n_splits=3, random_state=0, test_size=0.10
    ),
)

if __name__ == "__main__":
    BenchmarkRunner(benchmark).start()
