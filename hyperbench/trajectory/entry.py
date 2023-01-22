from dataclasses import dataclass
from typing import Union


@dataclass
class Entry:
    conf: dict
    loss: float
    at_iteration: Union[int, float]  # float only when budget multiplier was used
    at_time: any
    seeds: list[int]
