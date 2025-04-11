from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import OmegaConf, MISSING
from hydra import main as hydra_main
from dataclasses import dataclass
from dataclasses import field
from typing import Any, List, Dict
from hydra.utils import instantiate
import pprint
from dotenv import load_dotenv
import os
import random
import numpy as np
from greek.init_configs import init_configs
from greek.trainer.config import AwesomeAlignTrainer

load_dotenv() # for local vs on cluster coding.
init_configs() # necessary for hydra to discover configs.
OmegaConf.register_new_resolver("eval", eval)
cs = ConfigStore.instance()


@dataclass
class CustomKargoLauncherConfig(SlurmQueueConf): 
    """ https://hydra.cc/docs/1.3/plugins/submitit_launcher/ then go to github and look at config.py this is what I extend.
        to run things locally, use the option on launch `python run.py hydra/launcher=submitit_local`, 
        or in this case, without -m it launches things locally.
    """
    # submitit_folder: str = 
    # the default submitit_folder = "${hydra.sweep.dir}/.submitit/%j"
    # so reasonable and can't make it anything more reasonable it seems, because 
    # they launch with map_executor on the backend, which is the best for my 
    # parallel jobs, but prevents nicely putting the submitit loggs into more 
    # careful foldering. Perhaps in the future I can follow a per experiment 
    # foldering, and the specificity of the sweep.dir folder will be more useful 
    # to me.
    timeout_min: int = 2880 # 60 * 24 * 2
    cpus_per_task: int|None = 6 # type: ignore
    gpus_per_node: int|None = None
    tasks_per_node: int =  1
    mem_gb: int|None =  None
    nodes: int = 1
    _target_: str = "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    partition: str|None = "overcap" # kargo-lab
    qos: str|None = "short"
    exclude: str|None = "major,crushinator,nestor,voltron,xaea-12"
    additional_parameters: Dict[str, Any] = field(default_factory=lambda: {"gpus": "a40:1", "requeue": True})
    array_parallelism: int = 20
cs.store(name="custom_kargo_submitit", node=CustomKargoLauncherConfig, group="hydra/launcher")

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"trainer": "AwesomeAlignTrainer"},
        {"override hydra/launcher": os.getenv("GREEK_LAUNCHER", "custom_kargo_submitit")},
        # {"override hydra/sweeper": "optuna"}, # https://hydra.cc/docs/plugins/optuna_sweeper/
        # {"override hydra/sweeper/sampler": "random"}, 
        "_self_",
        ])
    trainer: AwesomeAlignTrainer = MISSING
    preprocessing: Any = MISSING
    run_type: str|None = None
    node_name: str = MISSING
    device: str = os.getenv('device', "cuda")
    output_dir: str = MISSING
    seed: int = 2
    # batch_size: int = 128
    # known issue: the multirun.yaml is saved to the sweep dir, and not the subdirs, so it is not saved! (don't think I will need this to be saved tho, and makes folders easier to read)
    hydra: Any = field(default_factory=lambda: {
        "sweep":{"dir": "greek_runs", 
                 "subdir": "${run_type}_${now:%Y-%m-%d}/${now:%H-%M-%S}_${hydra.job.num}" },
        "run":{"dir":  "greek_runs/${run_type}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
        # "sweeper": {"sampler": "random"},
    })
cs.store(name="RunConfig", node=RunConfig)

# "run_modifier" group: to use multiple just run_modifer=[mod1,mod2,...].
OfflineRunConfig = {"defaults": [{"override /logger@trainer.logger": "BasicPrintLogger"}]}
cs.store(name="OfflineRunConfig", node=OfflineRunConfig, group="run_modifier", package="_global_")

# Question: What if multiple values are rewritten? like num_workers is changed in multiple places?
# Answer: the last one defined is used.
DebugRunConfig = {"trainer": {"datasetloaders":{"num_workers": 0}},
                  "hydra": {
                      "sweep":{"dir": "greek_runs", 
                              "subdir": "debug_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
                      "run":{"dir":  "greek_runs/debug_${now:%Y-%m-%d}/${now:%H-%M-%S}"}}}
cs.store(name="DebugRunConfig", node=DebugRunConfig, group="run_modifier", package="_global_")

OverrideConfig = {"hydra": {"job":{"config":{"override_dirname":{"item_sep": "_"}}},
                            "sweep":{"dir": "greek_runs", 
                                    "subdir": "${hydra.job.override_dirname}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
                            "run":{"dir":  "greek_runs/${hydra.job.override_dirname}_${now:%Y-%m-%d}/${now:%H-%M-%S}"}}}
cs.store(name="OverrideConfig", node=OverrideConfig, group="run_modifier", package="_global_")

SupervisedRunConfig = {"defaults": [{"override /dataset@trainer.datasetloaders.train_dataset": "JaEnSupervisedAwesomeAlignDatasetTraining"},
                                    {"override /datasetmap@trainer.datasetloaders.val_datasets": "JaEnSupervisedAwesomeAlignDatasetsMapEval"}],
                       "trainer": {"datasetloaders":{"batch_size": 8},
                                   "max_epochs": 5,
                                   "model": {"train_supervised": True},
                                   "get_optimizer": {"lr": 1e-4},
                                   "log_every_n_steps": 20,
                                   "val_every_n_steps": 100,
                                   "val_plot_every_n_steps": 200},
                       "run_type": "supervised"}
cs.store(name="SupervisedRunConfig", node=SupervisedRunConfig, group="run_modifier", package="_global_")

ShortTrainRunConfig = {"defaults":  [{"override /dataset@trainer.datasetloaders.train_dataset": "JaEnUnsupervisedAwesomeAlignDatasetTraining"},
                                    {"override /datasetmap@trainer.datasetloaders.val_datasets": "JaEnSupervisedAwesomeAlignDatasetsMapEval"}],
                       "trainer": {"datasetloaders": {"batch_size": 8},
                                #    "max_epochs": 5,
                                #    "model": {"train_so": True},
                                   "max_steps": 20000,
                                   "get_optimizer": {"lr": 2e-5},
                                   "log_every_n_steps": 10,
                                   "val_every_n_steps": 1000,
                                   "val_plot_every_n_steps": 1000},
                       "run_type": "short_train"}
cs.store(name="ShortTrainRunConfig", node=ShortTrainRunConfig, group="run_modifier", package="_global_")

ShortTrainSORunConfig = {"defaults":  [{"/run_modifier": ["ShortTrainRunConfig"]}],
                       "trainer":  {"model": {"train_so": True}},
                       "run_type": "short_train_so"}
cs.store(name="ShortTrainSORunConfig", node=ShortTrainSORunConfig, group="run_modifier", package="_global_")

ShortTrainTLMRunConfig = {"defaults":  [{"/run_modifier": ["ShortTrainRunConfig"]}], # for some reason run_modifier must be a list.
                       "trainer":  {"model": {"train_tlm": True}},
                       "run_type": "short_train_tlm"}
cs.store(name="ShortTrainTLMRunConfig", node=ShortTrainTLMRunConfig, group="run_modifier", package="_global_")

ShortTrainMLMRunConfig = {"defaults":  [{"/run_modifier": ["ShortTrainRunConfig"]}], # for some reason run_modifier must be a list.
                       "trainer":  {"model": {"train_mlm": True}},
                       "run_type": "short_train_mlm"}
cs.store(name="ShortTrainMLMRunConfig", node=ShortTrainMLMRunConfig, group="run_modifier", package="_global_")

ShortTrainPSIRunConfig = {"defaults":  [{"/run_modifier": ["ShortTrainRunConfig"]}], # for some reason run_modifier must be a list.
                       "trainer":  {"model": {"train_psi": True}},
                       "run_type": "short_train_psi"}
cs.store(name="ShortTrainPSIRunConfig", node=ShortTrainPSIRunConfig, group="run_modifier", package="_global_")

ShortTrainAllRunConfig = {"defaults":  [{"/run_modifier": ["ShortTrainRunConfig"]}], # for some reason run_modifier must be a list.
                       "trainer":  {"model": {"train_psi": True,
                                              "train_mlm": True,
                                              "train_tlm": True, # this doesn't have full tlm enabled, so less tlm used.
                                              "train_so": True}},
                       "run_type": "short_train_all"}
cs.store(name="ShortTrainAllRunConfig", node=ShortTrainAllRunConfig, group="run_modifier", package="_global_")

ClassificationTrainRunConfig = {"defaults":[{"override /dataset@trainer.datasetloaders.train_dataset": "JaEnSupervisedAwesomeAlignDatasetTraining"},
                                            {"override /datasetmap@trainer.datasetloaders.val_datasets": "JaEnSupervisedAwesomeAlignDatasetsMapEval"}],
                       "trainer": { "datasetloaders":{"batch_size": 8},
                                    "max_epochs": 5,
                                    "get_optimizer": {"lr": 2e-5},
                                    "log_every_n_steps": 20,
                                    "val_every_n_steps": 100,
                                    "val_plot_every_n_steps": 200,
                                    "model": {
                                            "maskedlm": {"pretrained_model_name_or_path": "/nethome/jbjorner3/dev/diffusion-fun/text_diffusion/greek/awesome-align/greek/model_replicate_full_out_2"},
                                            # "train_mlm": True,
                                            # "mlm_weight": 1,
                                            # "train_tlm": True, # this doesn't have full tlm enabled, so less tlm used.
                                            # "train_tlm_full": True,
                                            # "tlm_weight": 1,
                                            "train_classification": True, # different from train_supervised.
                                            "classifier_threshold": 0.5,
                                            #   "train_so": True,
                                            }
                                    },
                       "run_type": "classification"}
cs.store(name="ClassificationTrainRunConfig", node=ClassificationTrainRunConfig, group="run_modifier", package="_global_")

FullTrainRunConfig = {"defaults":  [{"override /dataset@trainer.datasetloaders.train_dataset": "MultilingualUnsupervisedAwesomeAlignDatasetTraining"},
                                    {"override /datasetmap@trainer.datasetloaders.val_datasets": "nozhSupervisedAwesomeAlignDatasetsMapEval"}],
                       "trainer": {"datasetloaders": {"batch_size": 8},
                                #    "max_epochs": 5,
                                   "model": {"train_so": True,
                                             "train_psi": True,
                                             "train_mlm": True,
                                             "train_tlm": True,
                                             "train_tlm_full": True},
                                   "max_steps": 40000,
                                   "get_optimizer": {"lr": 2e-5},
                                   "log_every_n_steps": 100,
                                   "val_every_n_steps": 2000,
                                   "val_plot_every_n_steps": 2000},
                       "run_type": "full_train"}
cs.store(name="FullTrainRunConfig", node=FullTrainRunConfig, group="run_modifier", package="_global_")


@hydra_main(version_base=None, config_name='RunConfig')
def my_app(cfg: RunConfig) -> None:
    import torch
    # lazy import for fast hydra command line utility.

    # if cfg.device == "mps":
    #     assert torch.backends.mps.is_available(), "mps must be available for mps device spec"
    # elif cfg.device == "cuda":
    #     assert torch.cuda.is_available(), "cuda must be available for cuda device"
    # else:
    #     raise Exception(f"device {cfg.device} cannot be specified. No cpu because don't like slow on accident.")

    cfg.node_name = os.getenv("SLURMD_NODENAME", "NO_NODE_NAME_FOUND")
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    ppconfig = cfg.preprocessing
    cfg.trainer.logger.name = f"{cfg.run_type}"
    if cfg.trainer.model.mlm_probability != 0.15 or cfg.trainer.model.word_masking:
        cfg.trainer.logger.name += f"_mratio={cfg.trainer.model.mlm_probability}_wmask={cfg.trainer.model.word_masking}"
    if hasattr(cfg.trainer.model.maskedlm, "combine_layer") and cfg.trainer.model.maskedlm.combine_layer >= 0:  # type: ignore
        cfg.trainer.logger.name += f"_comb_exp={(cfg.trainer.model.maskedlm.combine_layer, cfg.trainer.model.maskedlm.expand_layer)}" # type: ignore
    if ppconfig.prob_combine != 0.0:
        cfg.trainer.logger.name += f"_pp_cdms={(ppconfig.prob_combine, ppconfig.prob_delete, ppconfig.prob_move, ppconfig.prob_swap)}"
    if cfg.trainer.model.supervised_weight != 1 or cfg.trainer.model.so_weight != 1 or cfg.trainer.model.mlm_weight != 1 or cfg.trainer.model.tlm_weight != 1 or cfg.trainer.model.psi_weight != 1:
        cfg.trainer.logger.name += f"_ws=({cfg.trainer.model.supervised_weight:.3},{cfg.trainer.model.so_weight:.3},{cfg.trainer.model.mlm_weight:.3},{cfg.trainer.model.tlm_weight:.3},{cfg.trainer.model.psi_weight:.3})"
    # if "max_softmax" in cfg.trainer.model.coverage_encouragement_type:
    #     cfg.trainer.logger.name += f"_maxCvgTempStartEnd={(cfg.trainer.model.max_softmax_temperature_start, cfg.trainer.model.max_softmax_temperature_end)}_cvgW={cfg.trainer.model.coverage_weight}_cvgType={cfg.trainer.model.coverage_encouragement_type}"
    if cfg.trainer.model.train_classification:
        if cfg.trainer.model.classifier._target_.endswith(".ConvClassifierNet"):
            cfg.trainer.logger.name += f"_cls={(cfg.trainer.model.classifier.hidden_dim, cfg.trainer.model.classifier.conv_hidden, cfg.trainer.model.classifier.dropout_rate)}"
        elif cfg.trainer.model.classifier._target_.endswith(".ClassifierNet"):
            cfg.trainer.logger.name += f"_cls_h={(cfg.trainer.model.classifier.hidden_dim)}"
        cfg.trainer.logger.name += f"_cls_w={cfg.trainer.model.classification_weight}_thresh={cfg.trainer.model.classifier_threshold}_btc={cfg.trainer.datasetloaders.batch_size}_lyr={cfg.trainer.model.layer_number}"
    if cfg.trainer.model.maskedlm.pretrained_model_name_or_path != "google-bert/bert-base-multilingual-cased":
        cfg.trainer.logger.name += f"_mod_b={cfg.trainer.model.maskedlm.pretrained_model_name_or_path.split('/')[-1]}"
    cfg.trainer.logger.name += f"_lr={cfg.trainer.get_optimizer.lr}_seed={cfg.seed}"
    # _cosSim={cfg.trainer.model.cosine_sim}_simTemp={cfg.trainer.model.sim_func_temp}_thresh={cfg.trainer.model.threshold}_divByLen={cfg.trainer.model.div_by_len}_entropyLoss={cfg.trainer.model.entropy_loss}
    # import ipdb; ipdb.set_trace()

    isMultirun = "num" in HydraConfig.get().job # type: ignore # for implicit debugging when launching a job without -m.  
    # cfg.datasetloaders.num_workers =  3 if not isMultirun else HydraConfig.get().launcher.cpus_per_task - 3
    cfg_for_logging = OmegaConf.to_container(cfg)
    seed = cfg.seed # the preprocessing that occurs in the dataset objects for corruptions need the seeds set for consistency.
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    trainer = instantiate(cfg.trainer)
    # using with wandb.init is a hack to get wandb to create a new run for every -m sweep. otherwise it concats them to one run.
    with trainer.logger.init(config=cfg_for_logging) as run:
        ret = trainer.fit(cfg)
    if isinstance(ret, dict) and "eval_jaen_AER" in ret:
        return ret["eval_jaen_AER"]



if __name__ == "__main__":
    my_app()