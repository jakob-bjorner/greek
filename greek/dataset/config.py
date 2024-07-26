from dataclasses import dataclass, field
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from typing import Any, List, Dict, Optional
import os


@dataclass
class AwesomeAlignDataset:
    _target_: str = "greek.dataset.dataset.AwesomeAlignDataset"
    tokenizer: Any = "${tokenizer}"
    src_tgt_file: str = MISSING
    gold_file: Optional[str] = MISSING
    gold_one_index: bool = True
    
    # TODO: mess with using different weightings on possible aligns in the supervised setting or something like this?
    ignore_possible_alignments: bool = False 

DATA_FOLDER  = os.path.abspath(os.path.join(__file__, "../../../data"))

ConfigStore.instance().store(name="MultilingualUnsupervisedAwesomeAlignDatasetTraining", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_training_data/multilingual_data_nozh.src-tgt", gold_file=None), group="dataset")
ConfigStore.instance().store(name="JaEnUnsupervisedAwesomeAlignDatasetTraining", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_training_data/jaen.src-tgt", gold_file=None), group="dataset")

ConfigStore.instance().store(name="JaEnSupervisedAwesomeAlignDatasetEval", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_eval_data/jaen.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_eval_data/jaen.gold", gold_one_index=False), group="dataset")
# ConfigStore.instance().store(name="JaEnUnsupervisedAwesomeAlignDatasetEval", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_eval_data/old/old_jaen.src-tgt", gold_file=None, gold_one_index=False), group="dataset")

# TODO: add support for multiple seperate eval and test sets, so can train all at once with a fraction of all languages and test quickly in one command.
ConfigStore.instance().store(name="DeEnSupervisedAwesomeAlignDatasetTest", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/deen.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/deen.gold"), group="dataset")
ConfigStore.instance().store(name="EnFrSupervisedAwesomeAlignDatasetTest", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/enfr.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/enfr.gold"), group="dataset")
ConfigStore.instance().store(name="RoEnSupervisedAwesomeAlignDatasetTest", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/roen.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/roen.gold"), group="dataset")
ConfigStore.instance().store(name="JaEnSupervisedAwesomeAlignDatasetTest", node=AwesomeAlignDataset(src_tgt_file=f"{DATA_FOLDER}/awesome_test_examples/jaen.src-tgt", gold_file=f"{DATA_FOLDER}/awesome_test_examples/jaen.gold", gold_one_index=True), group="dataset")
