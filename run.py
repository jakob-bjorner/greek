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
load_dotenv()
init_configs()
OmegaConf.register_new_resolver("eval", eval)

@dataclass
class CustomKargoLauncherConfig(SlurmQueueConf): 
    """ https://hydra.cc/docs/1.3/plugins/submitit_launcher/ then go to github and look at config.py this is what I extend.
        to run things locally, use the option on launch `python run.py hydra/launcher=submitit_local`, 
        or in this case, without -m it launches things locally.
    """
    submitit_folder: str = "${hydra.sweep.dir}/.submitit/%j"
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
    additional_parameters: Dict[str, str] = field(default_factory=lambda: {"gpus": "${device_type}:1"})
    array_parallelism: int = 20

@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        # {"model": "VAEVisionModelConfig"},
        # {"datasetloaders": "Cifar10DatasetLoadersConfig"},
        # {"trainer": "VAETrainerConfig"},
        {"override hydra/launcher": os.getenv("GREEK_LAUNCHER", "custom_kargo_submitit")},
        "_self_",
        ])
    # datasetloaders: Any = MISSING
    # model: Any = MISSING
    # trainer: Any = MISSING
    project_name: str = "greek"
    run_name: str = MISSING
    node_name: str = MISSING
    device: str = "cuda"
    output_dir: str = MISSING
    # batch_size: int = 128
    device_type: str = "a40"
    hydra: Any = field(default_factory=lambda: {
        "job":{"config":{"override_dirname":{"item_sep": "_"}}},
        "sweep":{"dir": "greek_runs", 
                 "subdir": "${hydra.job.override_dirname}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
        "run":{"dir":  "greek_runs/${hydra.job.override_dirname}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
    })

cs = ConfigStore.instance()
cs.store(name="custom_kargo_submitit", node=CustomKargoLauncherConfig, group="hydra/launcher")
cs.store(name="basic_config", node=RunConfig)


@hydra_main(version_base=None, config_name='basic_config')
def my_app(cfg: RunConfig) -> None:
    # lazy import for fast hydra command line utility.
    import wandb
    import torch

    cfg.node_name = os.getenv("SLURMD_NODENAME", "could be mac?")
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    # encoder_model_name = cfg.model.encoder_model._target_.split('.')[-1]
    # decoder_model_name = cfg.model.decoder_model._target_.split('.')[-1]
    # cfg.run_name = f"pDim={cfg.model.prior_dim}_lr={cfg.trainer.encoder_lr}_en={encoder_model_name}_de={decoder_model_name}_bKl={cfg.model.beta}_ep={cfg.trainer.epochs}_bs={cfg.batch_size}"

    isMultirun = "num" in HydraConfig.get().job # type: ignore # for implicit debugging when launching a job without -m.
    # cfg.datasetloaders.num_workers =  3 if not isMultirun else HydraConfig.get().launcher.cpus_per_task - 3
    wandb_cfg = OmegaConf.to_container(cfg)
    print(pprint.saferepr(wandb_cfg))
    if isMultirun:
        with wandb.init(project=cfg.project_name, name=cfg.run_name, config=wandb_cfg) as run:  # type: ignore # don't know if this really does anything, but with regular init the jobs are all in one, and this fixes it!
            wandb.define_metric("custom_step")
            wandb.define_metric("evaluate_*", step_metric='custom_step') # allows for asynchronous logging of eval events.
            # trainer = instantiate(cfg.trainer, logger=wandb)
            # trainer.train()
    else:
        class BasicPrintLogger:
            def log(self, *args, **kwargs):
                print(args, kwargs)
        import ipdb; ipdb.set_trace() 
        # trainer = instantiate(cfg.trainer, logger=BasicPrintLogger())
        # trainer.train()


if __name__ == "__main__":
    my_app()