from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class Trainer:
    _target_: str = MISSING

ConfigStore.instance().store(name="trainer", node=Trainer, group="trainer")