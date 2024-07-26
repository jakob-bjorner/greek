from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import List, Any 

@dataclass
class BasicMaskedLM:
    _target_: str = MISSING

@dataclass
class HFMaskedLM:
    _target_: str = "transformers.AutoModelForMaskedLM.from_pretrained"
    pretrained_model_name_or_path: str = MISSING

# BaseMaskedLM is intended to be used for when a model has been trained and you should just be able to load it's checkpoint and continue training from there.
ConfigStore.instance().store(name="BaseMaskedLM", node=HFMaskedLM, group="maskedlm") 
# PreTrainedBERTMaskedLM just loads the same bert model as awesome align trained from.
ConfigStore.instance().store(name="PreTrainedBERTMaskedLM", node=HFMaskedLM(pretrained_model_name_or_path="google-bert/bert-base-multilingual-cased"), group="maskedlm")
# DummyMaskedLM for fast loading and debugging without the model.
ConfigStore.instance().store(name="DummyMaskedLM", node=BasicMaskedLM(_target_="greek.model.model.DummyEncoder"), group="maskedlm") 

@dataclass
class Aligner:
    defaults: List[Any] = field(default_factory=lambda: [
        {"/maskedlm": "PreTrainedBERTMaskedLM"},
        "_self_",
        ])
    maskedlm: BasicMaskedLM = MISSING # important that the type be Any, because the type will be enforced in the application. ie need all the values defined for this type. encoder is flexible due to basic configs for debugging.
    tokenizer: Any = "${tokenizer}"
    device: str = "${device}"
    layer_number: int = 7
    threshold: float = 0.001 # default Awesome-align threshold.
    train_supervised: bool = False
    train_so: bool = False
    train_psi: bool = False
    train_mlm: bool = False
    train_tlm: bool = False
    mlm_probability: float = 0.15 # default for bert/roberta
    # train_co
    _target_: str = MISSING

ConfigStore.instance().store(name="AwesomeAligner", node=Aligner(_target_="greek.model.model.AwesomeAligner"), group="model")
