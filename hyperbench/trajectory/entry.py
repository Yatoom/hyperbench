from dataclasses import dataclass


@dataclass
class Entry:
    conf: dict
    loss: float
    at_iteration: int
    at_time: any
    seeds: list[int]
