from dataclasses import dataclass
from typing import Union


@dataclass(eq=False)
class Entry:
    conf: dict
    loss: float
    at_iteration: Union[int, float]  # float only when budget multiplier was used
    at_time: any
    seeds: list[int]

    def __eq__(self, other):
        cond1 = self.conf == other.conf
        cond2 = self.loss == other.loss
        cond3 = self.at_iteration == other.at_iteration
        cond4 = self.seeds == other.seeds
        return cond1 and cond2 and cond3 and cond4
