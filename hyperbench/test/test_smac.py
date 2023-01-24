from rich.progress import TextColumn, TimeRemainingColumn, TimeElapsedColumn, BarColumn, MofNCompleteColumn, Progress
from sklearn.model_selection import ShuffleSplit
from smac.facade.roar_facade import ROAR
from smac.facade.smac_hpo_facade import SMAC4HPO

from hyperbench.optimizers import SMACBasedOptimizer
from hyperbench.provider import OpenMLProvider
from hyperbench.target_algorithms import XGBoost

# Import SMAC
# 49
data = OpenMLProvider(49).data
target = XGBoost()
train_test_splits = ShuffleSplit(n_splits=3, random_state=0, test_size=0.10)
scoring = "balanced_accuracy"
budget = 10
progress = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(),
                            TimeRemainingColumn(), TimeElapsedColumn())
loop_iterations = progress.add_task("Evaluations", total=budget)

roar = SMACBasedOptimizer(ROAR, "roar_x2", budget_multiplier=1)
smac = SMACBasedOptimizer(SMAC4HPO, "smac", budget_multiplier=1)
tae = target.get_config_evaluator(data, train_test_splits, scoring, progress, loop_iterations)

with progress as progress:
    roar.initialize(tae, 1, data, budget, False, target)
    roar.search()
    roar.get_trajectory()
    print()