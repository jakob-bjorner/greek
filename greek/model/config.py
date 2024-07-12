from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

@dataclass
class BaseModel:
    ...