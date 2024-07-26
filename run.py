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
    exclude: str|None = "major,crushinator,nestor,voltron"
    additional_parameters: Dict[str, str] = field(default_factory=lambda: {"gpus": "a40:1"})
    array_parallelism: int = 20
cs.store(name="custom_kargo_submitit", node=CustomKargoLauncherConfig, group="hydra/launcher")

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"trainer": "AwesomeAlignTrainer"},
        {"override hydra/launcher": os.getenv("GREEK_LAUNCHER", "custom_kargo_submitit")},
        "_self_",
        ])
    trainer: AwesomeAlignTrainer = MISSING
    run_type: str|None = None
    node_name: str = MISSING
    device: str = os.getenv('device', "cuda")
    output_dir: str = MISSING
    # batch_size: int = 128
    # known issue: the multirun.yaml is saved to the sweep dir, and not the subdirs, so it is not saved! (don't think I will need this to be saved tho, and makes folders easier to read)
    hydra: Any = field(default_factory=lambda: {
        "job":{"config":{"override_dirname":{"item_sep": "_"}}},
        "sweep":{"dir": "greek_runs", 
                 "subdir": "${hydra.job.override_dirname}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
        "run":{"dir":  "greek_runs/${hydra.job.override_dirname}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
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
                              "subdir": "$debug_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
                      "run":{"dir":  "greek_runs/$debug_${now:%Y-%m-%d}/${now:%H-%M-%S}"}}}
cs.store(name="DebugRunConfig", node=DebugRunConfig, group="run_modifier", package="_global_")

SupervisedRunConfig = {"defaults": [{"override /dataset@trainer.datasetloaders.train_dataset": "JaEnSupervisedAwesomeAlignDatasetEval"},
                                    {"override /dataset@trainer.datasetloaders.val_dataset": "JaEnSupervisedAwesomeAlignDatasetTest"}],
                       "trainer": {"datasetloaders":{"batch_size": 8},
                                   "max_epochs": 5,
                                   "model": {"train_supervised": True},
                                   "get_optimizer": {"lr": 1e-4},
                                   "log_every_n_steps": 10,
                                   "val_every_n_steps": 100},
                       "run_type": "supervised"}
cs.store(name="SupervisedRunConfig", node=SupervisedRunConfig, group="run_modifier", package="_global_")

ShortTrainRunConfig = {"defaults":  [{"override /dataset@trainer.datasetloaders.train_dataset": "JaEnUnsupervisedAwesomeAlignDatasetTraining"},
                                    {"override /dataset@trainer.datasetloaders.val_dataset": "JaEnSupervisedAwesomeAlignDatasetEval"}],
                       "trainer": {"datasetloaders": {"batch_size": 8},
                                #    "max_epochs": 5,
                                #    "model": {"train_so": True},
                                   "max_steps": 20000,
                                   "get_optimizer": {"lr": 2e-5},
                                   "log_every_n_steps": 10,
                                   "val_every_n_steps": 1000},
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



@hydra_main(version_base=None, config_name='RunConfig')
def my_app(cfg: RunConfig) -> None:
    # lazy import for fast hydra command line utility.
    import torch

    # if cfg.device == "mps":
    #     assert torch.backends.mps.is_available(), "mps must be available for mps device spec"
    # elif cfg.device == "cuda":
    #     assert torch.cuda.is_available(), "cuda must be available for cuda device"
    # else:
    #     raise Exception(f"device {cfg.device} cannot be specified. No cpu because don't like slow on accident.")

    cfg.node_name = os.getenv("SLURMD_NODENAME", "NO_NODE_NAME_FOUND")
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    cfg.trainer.logger.name = f"{cfg.run_type}_ep={cfg.trainer.max_epochs}_bs={cfg.trainer.datasetloaders.batch_size}_lr={cfg.trainer.get_optimizer.lr}"

    isMultirun = "num" in HydraConfig.get().job # type: ignore # for implicit debugging when launching a job without -m.
    # cfg.datasetloaders.num_workers =  3 if not isMultirun else HydraConfig.get().launcher.cpus_per_task - 3
    cfg_for_logging = OmegaConf.to_container(cfg)

    trainer = instantiate(cfg.trainer)
    # using with wandb.init is a hack to get wandb to create a new run for every -m sweep. otherwise it concats them to one run.
    with trainer.logger.init(config=cfg_for_logging) as run:
        trainer.fit()


if __name__ == "__main__":
    my_app()