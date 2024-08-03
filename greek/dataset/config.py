from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List, Dict, Optional, Tuple
from collections.abc import Callable
import os



@dataclass
class AwesomeAlignDataset:
    defaults: List[Any] = field(default_factory=lambda:[
        {"/preprocessing": "identity_preprocessing"},
        "_self_",
        ])
    _target_: str = "greek.dataset.dataset.AwesomeAlignDataset"
    tokenizer: Any = "${tokenizer}"
    src_tgt_file: str = MISSING
    gold_file: Optional[str] = MISSING
    gold_one_index: bool = True
    
    # TODO: mess with using different weightings on possible aligns in the supervised setting or something like this?
    ignore_possible_alignments: bool = False 
    preprocessing: Any = MISSING
@dataclass 
class AwesomeAlignDatasetMultilingualTraining:
    defaults: List[Any] = field(default_factory=lambda:[
        {"/preprocessing": "identity_preprocessing"},
        "_self_",
        ])
    _target_: str = "greek.dataset.dataset.AwesomeAlignDatasetMultilingualTraining"
    tokenizer: Any = "${tokenizer}"
    src_tgt_enfr_file: str = MISSING
    src_tgt_roen_file: str = MISSING
    src_tgt_deen_file: str = MISSING
    src_tgt_jaen_file: str = MISSING
    len_per_lang: int = 200000
    preprocessing: Any = MISSING
@dataclass
class AwesomeAlignDatasetsMap:
    _target_: str = "greek.dataset.dataset.AwesomeAlignDatasetsMap"
    defaults: List[Any] = field(default_factory=lambda: [
        "_self_",
        ])
    enfr_dataset: Optional[AwesomeAlignDataset] = None
    deen_dataset: Optional[AwesomeAlignDataset] = None
    roen_dataset: Optional[AwesomeAlignDataset] = None
    jaen_dataset: Optional[AwesomeAlignDataset] = None

DATA_FOLDER  = os.path.abspath(os.path.join(__file__, "../../../data"))

# "trainer.datasetloaders.train_dataset.preprocessing={prob_combine: 0.5, prob_delete: 0.2, prob_swap: 0.2}"
ConfigStore.instance().store(name="identity_preprocessing", node={"_target_": "greek.dataset.dataset.preprocessing", "prob_combine":0.0, "prob_delete": 0.0, "prob_swap": 0.0, "_partial_": True}, group="preprocessing")

# ConfigStore.instance().store(name="MultilingualUnsupervisedAwesomeAlignDatasetTraining", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_training_data/multilingual_data_nozh.src-tgt", gold_file=None), group="dataset")
ConfigStore.instance().store(name="MultilingualUnsupervisedAwesomeAlignDatasetTraining", node=AwesomeAlignDatasetMultilingualTraining(src_tgt_enfr_file=f"{DATA_FOLDER}/awesome_training_data/enfr.src-tgt",
                                                                                                                                      src_tgt_roen_file=f"{DATA_FOLDER}/awesome_training_data/roen.src-tgt",
                                                                                                                                      src_tgt_deen_file=f"{DATA_FOLDER}/awesome_training_data/deen.src-tgt",
                                                                                                                                      src_tgt_jaen_file=f"{DATA_FOLDER}/awesome_training_data/jaen.src-tgt"), group="dataset")
ConfigStore.instance().store(name="JaEnUnsupervisedAwesomeAlignDatasetTraining", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_training_data/jaen.src-tgt", gold_file=None), group="dataset")
ConfigStore.instance().store(name="JaEnSupervisedAwesomeAlignDatasetTraining", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_training_data/jaen_train_past100.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_training_data/jaen_train_past100.gold", gold_one_index=False), group="dataset")

ConfigStore.instance().store(name="JaEnSupervisedAwesomeAlignDatasetEval", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_eval_data/jaen_eval_first100.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_eval_data/jaen_eval_first100.gold", gold_one_index=False), group="dataset")
ConfigStore.instance().store(name="DeEnSupervisedAwesomeAlignDatasetEval", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_eval_data/deen_eval_first100.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_eval_data/deen_eval_first100.gold", gold_one_index=True), group="dataset")
ConfigStore.instance().store(name="EnFrSupervisedAwesomeAlignDatasetEval", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_eval_data/enfr_eval_first100.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_eval_data/enfr_eval_first100.gold", gold_one_index=True), group="dataset")
ConfigStore.instance().store(name="RoEnSupervisedAwesomeAlignDatasetEval", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_eval_data/roen_eval_first50.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_eval_data/roen_eval_first50.gold", gold_one_index=True), group="dataset")
ConfigStore.instance().store(name="nozhSupervisedAwesomeAlignDatasetsMapEval", node=AwesomeAlignDatasetsMap(defaults=["_self_",
                                                                                                                      {"/dataset@jaen_dataset": "JaEnSupervisedAwesomeAlignDatasetEval"}, 
                                                                                                                      {"/dataset@deen_dataset": "DeEnSupervisedAwesomeAlignDatasetEval"},
                                                                                                                      {"/dataset@enfr_dataset": "EnFrSupervisedAwesomeAlignDatasetEval"},
                                                                                                                      {"/dataset@roen_dataset": "RoEnSupervisedAwesomeAlignDatasetEval"}]), group="datasetmap")
ConfigStore.instance().store(name="JaEnSupervisedAwesomeAlignDatasetsMapEval", node=AwesomeAlignDatasetsMap(defaults=["_self_", {"/dataset@jaen_dataset": "JaEnSupervisedAwesomeAlignDatasetEval"}]), group="datasetmap")

# ConfigStore.instance().store(name="JaEnUnsupervisedAwesomeAlignDatasetEval", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_eval_data/old/old_jaen.src-tgt", gold_file=None, gold_one_index=False), group="dataset")

ConfigStore.instance().store(name="DeEnSupervisedAwesomeAlignDatasetTestFull", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/deen.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/deen.gold"), group="dataset")
ConfigStore.instance().store(name="EnFrSupervisedAwesomeAlignDatasetTestFull", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/enfr.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/enfr.gold"), group="dataset")
ConfigStore.instance().store(name="RoEnSupervisedAwesomeAlignDatasetTestFull", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/roen.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/roen.gold"), group="dataset")
ConfigStore.instance().store(name="JaEnSupervisedAwesomeAlignDatasetTest", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/jaen.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/jaen.gold", gold_one_index=True), group="dataset")
ConfigStore.instance().store(name="DeEnSupervisedAwesomeAlignDatasetTest", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/deen_test_past100.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/deen_test_past100.gold", gold_one_index=True), group="dataset")
ConfigStore.instance().store(name="EnFrSupervisedAwesomeAlignDatasetTest", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/enfr_test_past100.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/enfr_test_past100.gold", gold_one_index=True), group="dataset")
ConfigStore.instance().store(name="RoEnSupervisedAwesomeAlignDatasetTest", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/roen_test_past50.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/roen_test_past50.gold", gold_one_index=True), group="dataset")
ConfigStore.instance().store(name="nozhSupervisedAwesomeAlignDatasetsMapTest", node=AwesomeAlignDatasetsMap(defaults=["_self_",
                                                                                                                      {"/dataset@jaen_dataset": "JaEnSupervisedAwesomeAlignDatasetTest"},
                                                                                                                      {"/dataset@deen_dataset": "DeEnSupervisedAwesomeAlignDatasetTest"},
                                                                                                                      {"/dataset@enfr_dataset": "EnFrSupervisedAwesomeAlignDatasetTest"},
                                                                                                                      {"/dataset@roen_dataset": "RoEnSupervisedAwesomeAlignDatasetTest"}]), group="datasetmap")
