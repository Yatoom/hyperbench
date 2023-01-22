from dataclasses import dataclass

from ConfigSpace import ConfigurationSpace


@dataclass
class TargetAlgorithm:
    name: str
    model: any
    preset_params: dict
    config_space: ConfigurationSpace
    is_deterministic: bool

    def initialize(self, seed, **config):
        return self.model(random_state=seed, **self.preset_params, **config)
