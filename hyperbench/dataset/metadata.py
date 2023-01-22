from dataclasses import dataclass


@dataclass
class Metadata:
    id: str
    name: str
    categorical: list[int]
    numeric: list[int]
    n_rows: int
    n_columns: int
    n_classes: int
    n_missing: int
