from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List
from greek.model.config import Aligner
from greek.logger.config import BaseLogger
from greek.datasetloaders.config import AwesomeAlignDatasetLoaders

# TODO: move this to the model.py as it is directly relevant there like lightning
@dataclass
class OptimizerCallable:
    _target_: str = "greek.trainer.trainer.get_optimizer"
    custom: str = ""
    lr: float = 0.001
    weight_decay: float = 0.0 # this is different from default adamW which uses 0.01
    eps: float = 1e-8
    _partial_: bool = True
ConfigStore.instance().store(name="BERTAdamW", node=OptimizerCallable(custom="BERTAdamW"), group="optimizer")

@dataclass
class SchedulerCallable:
    _target_: str = "greek.trainer.trainer.get_scheduler"
    custom: str = ""
    warmup_steps: int = MISSING
    # num_training_steps: int must be given in the init
    _partial_: bool = True
ConfigStore.instance().store(name="BERT_linear_warmup", node=SchedulerCallable(custom="linear_warmup", warmup_steps=0), group="scheduler")


@dataclass
class AwesomeAlignTrainer:
    defaults: List[Any] = field(default_factory=lambda : [
        {"/model": "AwesomeAligner"},
        {"/datasetloaders": "AwesomeAlignDatasetLoaders"},
        {"/logger": "WandBLogger"},
        {"/optimizer@get_optimizer": "BERTAdamW"},
        {"/scheduler@get_scheduler": "BERT_linear_warmup"},
        "_self_",
        ])
    _target_: str = "greek.trainer.trainer.AwesomeAlignTrainer"
    model: Aligner = MISSING
    datasetloaders: AwesomeAlignDatasetLoaders = MISSING
    logger: BaseLogger = MISSING
    get_optimizer: OptimizerCallable = MISSING
    get_scheduler: SchedulerCallable = MISSING
    max_grad_norm: float = 1.0
    device: str = "${device}"
    max_steps: int = -1 # non negative to set a limit. Must set one of these
    max_epochs: float = -1.0 # non negative to set a limit
    log_every_n_steps: int = 500
    val_every_n_steps: int = 2000
    val_plot_every_n_steps: int = 2000
    output_dir: str = "${hydra:runtime.output_dir}"
    # seed: int = 0 # for numbers 0 or lower, the seed is random. This is for easy testing.
ConfigStore.instance().store(name="AwesomeAlignTrainer", node=AwesomeAlignTrainer, group="trainer")


