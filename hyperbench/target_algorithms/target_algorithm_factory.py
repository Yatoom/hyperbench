from abc import abstractmethod

from hyperbench.target_algorithms.target_algorithm import TargetAlgorithm


class TargetAlgorithmFactory:
    @staticmethod
    @abstractmethod
    def build() -> TargetAlgorithm:
        pass
