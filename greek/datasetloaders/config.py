from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List, Dict, Optional
from greek.dataset.config import AwesomeAlignDataset, AwesomeAlignDatasetsMap

@dataclass
class HFTokenizer:
    _target_: str = "transformers.AutoTokenizer.from_pretrained"
    pretrained_model_name_or_path: str = MISSING

@dataclass
class CustTokenizer: # this for debugging to see how many times the tokenizer is initialized.
    _target_: str = "greek.datasetloaders.datasetloaders.CustTokenizer"
    pretrained_model_name_or_path: str = MISSING
ConfigStore.instance().store(name="BERTTokenizer", node=HFTokenizer(pretrained_model_name_or_path="google-bert/bert-base-multilingual-cased"), group="tokenizer")
ConfigStore.instance().store(name="CustTokenizer", node=CustTokenizer(pretrained_model_name_or_path="google-bert/bert-base-multilingual-cased"), group="tokenizer")

@dataclass
class AwesomeAlignDatasetLoaders:
    defaults: List[Any] = field(default_factory=lambda: [
        # I have to place the tokenizer somewhere, and it can't be in this class or a circular dependancy occurs?
        # TODO: figure out how to make the tokenizer only be instantiated once, and have it passed to all objects as the same instance.
        {"/tokenizer@_global_.tokenizer": "BERTTokenizer"},
        {"/dataset@train_dataset": "MultilingualUnsupervisedAwesomeAlignDatasetTraining"},
        {"/datasetmap@val_datasets": "nozhSupervisedAwesomeAlignDatasetsMapEval"},
        {"/datasetmap@test_datasets": "nozhSupervisedAwesomeAlignDatasetsMapTest"},
        # "val_dataset", # the val dataset has the same possible ideas as the test dataset
        # "test_dataset", # the test dataset should be one per language? so really should be a potential list of tests to run.
        "_self_",
    ])
    _target_: str = "greek.datasetloaders.datasetloaders.AwesomeAlignDatasetLoaders"
    tokenizer: Any = "${tokenizer}"
    train_dataset: Any = MISSING
    val_datasets: AwesomeAlignDatasetsMap = MISSING
    test_datasets: AwesomeAlignDatasetsMap = MISSING
    batch_size: int = 8
    num_workers: int = 3 # change this for debugging.
    pin_memory: bool = True
    pin_memory_device: str = "${device}"

ConfigStore.instance().store(name="AwesomeAlignDatasetLoaders", node=AwesomeAlignDatasetLoaders, group="datasetloaders")