from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import List, Any, Optional

# @dataclass
# class BasicMaskedLM:
#     _target_: str = MISSING

@dataclass
class HFMaskedLM:
    _target_: str = "transformers.AutoModelForMaskedLM.from_pretrained"
    pretrained_model_name_or_path: str = MISSING

@dataclass
class PreTrainedBERTMaskedLMWithLSTM:
    _target_: str = "greek.model.model.customBertinit"
    pretrained_model_name_or_path: str = "google-bert/bert-base-multilingual-cased"
    combine_layer: int = MISSING
    expand_layer: int = MISSING

# BaseMaskedLM is intended to be used for when a model has been trained and you should just be able to load it's checkpoint and continue training from there.
ConfigStore.instance().store(name="BaseMaskedLM", node=HFMaskedLM, group="maskedlm") 
# PreTrainedBERTMaskedLM just loads the same bert model as awesome align trained from.
ConfigStore.instance().store(name="PreTrainedBERTMaskedLMWithLSTM", node=PreTrainedBERTMaskedLMWithLSTM, group="maskedlm")
ConfigStore.instance().store(name="PreTrainedBERTMaskedLM", node=PreTrainedBERTMaskedLMWithLSTM(pretrained_model_name_or_path="google-bert/bert-base-multilingual-cased", combine_layer=-1, expand_layer=-1), group="maskedlm")
# DummyMaskedLM for fast loading and debugging without the model.
# ConfigStore.instance().store(name="DummyMaskedLM", node=BasicMaskedLM(_target_="greek.model.model.DummyEncoder"), group="maskedlm") 

@dataclass
class ClassifierNetConfig:
    _target_: str = "greek.model.model.ClassifierNet"
    layer_norm: bool = False
    hidden_dim: int = 768*2

@dataclass
class ConvClassifierNetConfig:
    _target_: str = "greek.model.model.ConvClassifierNet"
    hidden_dim: int = 768*2
    conv_hidden: int = 64
    dropout_rate: float = 0.0

ConfigStore.instance().store(name="ClassifierNetConfig", node=ClassifierNetConfig, group="classifier")
ConfigStore.instance().store(name="ConvClassifierNetConfig", node=ConvClassifierNetConfig, group="classifier")

@dataclass
class Aligner:
    defaults: List[Any] = field(default_factory=lambda: [
        {"/maskedlm": "PreTrainedBERTMaskedLM"},
        {"/classifier": "ClassifierNetConfig"},
        "_self_",
        ])
    maskedlm: PreTrainedBERTMaskedLMWithLSTM = MISSING # important that the type be Any, because the type will be enforced in the application. ie need all the values defined for this type. encoder is flexible due to basic configs for debugging.
    classifier: Any = MISSING
    tokenizer: Any = "${tokenizer}"
    device: str = "${device}"
    layer_number: int = 7
    threshold: float = 0.001 # default Awesome-align threshold.
    train_supervised: bool = False
    train_so: bool = False
    train_psi: bool = False
    train_mlm: bool = False
    train_tlm: bool = False
    train_tlm_full: bool = False
    train_classification: bool = False
    mlm_probability: float = 0.15 # default for bert/roberta
    supervised_weight: float = 1
    so_weight: float = 1
    tlm_weight: float = 1
    mlm_weight: float = 1
    psi_weight: float = 1
    classification_weight: float = 1
    
    word_masking: bool = False
    entropy_loss: bool = False
    div_by_len: bool = True
    cosine_sim: bool = False
    sim_func_temp: float = 1.0 # 5 now 1/25=0.04, 10 now 1/100=0.01
    coverage_encouragement_type: str = "mse_softmax" # eventually with schedules, and other types "max_softmax"
    max_softmax_temperature_start: float = 1.0
    max_softmax_temperature_end: float = 1.0
    coverage_weight: float = 0.0
    classifier_threshold: float = 0.5

    # train_co
    _target_: str = MISSING

ConfigStore.instance().store(name="AwesomeAligner", node=Aligner(_target_="greek.model.model.AwesomeAligner"), group="model")
