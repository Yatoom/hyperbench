import openml
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from smac.facade.roar_facade import ROAR
from smac.facade.smac_hpo_facade import SMAC4HPO

from hyperbench.benchmark import BenchmarkConfig
from hyperbench.benchmark import BenchmarkRunner
from hyperbench.provider import OpenMLProvider
from hyperbench.optimizers import SMACBasedOptimizer
from hyperbench.target_algorithms import CatboostFactory
from hyperbench.target_algorithms import RandomForestFactory
from hyperbench.transformer import SimpleTransformer

tasks = openml.study.get_suite(99).tasks
# 0	CIFAR_10	        167124	93.8039
# 1	Devnagari-Script	167121	26.7153
# 2	Fashion-MNIST	    146825	15.4337
# 3	mnist_784	        3573	7.6295
# 4	numerai28.6	        167120	3.5865
# 5	har	                14970	2.6967
# 6	isolet	            3481	2.5220
# 7	electricity	        219	    1.8605
# 8	mfeat-factors	    12	    1.2955
tasks = [task for task in tasks if task not in [167124, 167121, 146825, 3573, 167120, 14970, 3481, 219, 12]]

benchmark = BenchmarkConfig(
    budget=60,
    time_based=True,
    transformer=SimpleTransformer(),
    output_folder="results/",
    scoring="balanced_accuracy", # make_scorer(roc_auc_score, multi_class='ovo'

    seeds=[2268061101, 2519249986, 338403738],
    target_algorithms=[CatboostFactory.build(), RandomForestFactory.build()],
    datasets=[OpenMLProvider(task).data for task in tasks],
    optimizers=[
        # SMACBasedOptimizer(ROAR, "roar_x1", budget_multiplier=1)
        SMACBasedOptimizer(ROAR, "roar_x2", budget_multiplier=2),
        SMACBasedOptimizer(SMAC4HPO, "smac", budget_multiplier=1)
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
