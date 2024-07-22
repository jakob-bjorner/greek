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

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"trainer": "AwesomeAlignTrainer"},
        {"override hydra/launcher": os.getenv("GREEK_LAUNCHER", "custom_kargo_submitit")},
        "_self_",
        ])
    is_offline: bool = True # changes logging, and what else? Should I just be creating a seperate config?
    trainer: AwesomeAlignTrainer = MISSING
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
@dataclass
class OfflineRunConfig(RunConfig):
    defaults: List[Any] = field(default_factory=lambda:[
        "RunConfig",
        {"override /logger@trainer.logger": "BasicPrintLogger"},
        "_self_",
    ])
    trainer: Any = field(default_factory=lambda:{"datasetloaders":{"num_workers": 0}})

cs = ConfigStore.instance()
cs.store(name="custom_kargo_submitit", node=CustomKargoLauncherConfig, group="hydra/launcher")
cs.store(name="RunConfig", node=RunConfig)
cs.store(name="OfflineRunConfig", node=OfflineRunConfig)


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
    # cfg.trainer.logger.name = f"pDim={cfg.model.prior_dim}_lr={cfg.trainer.encoder_lr}_en={encoder_model_name}_de={decoder_model_name}_bKl={cfg.model.beta}_ep={cfg.trainer.epochs}_bs={cfg.batch_size}"

    isMultirun = "num" in HydraConfig.get().job # type: ignore # for implicit debugging when launching a job without -m.
    # cfg.datasetloaders.num_workers =  3 if not isMultirun else HydraConfig.get().launcher.cpus_per_task - 3
    cfg_for_logging = OmegaConf.to_container(cfg)

    trainer = instantiate(cfg.trainer)
    # using with wandb.init is a hack to get wandb to create a new run for every -m sweep. otherwise it concats them to one run.
    with trainer.logger.init(config=cfg_for_logging) as run:
        trainer.fit()


if __name__ == "__main__":
    my_app()