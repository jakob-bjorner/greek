#%%
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from greek.model.model import AwesomeAligner, CollateFnReturn, AwesomeAlignerValMetrics
from greek.logger.logger import BaseLogger
from greek.datasetloaders.datasetloaders import AwesomeAlignDatasetLoaders
from typing import Any, Callable, Iterable
import random
import numpy as np
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
import tqdm
import os

# torch.optim.lr_scheduler.ChainedScheduler()
def get_optimizer(custom, **kwargs) -> Optimizer:
    if custom == "BERTAdamW" :
        # the adamW is an optimizer hyper specific to BERT, and in the future, might consider adding more flexible optimizers, but all will likely require strange codings from the config file.
        # Could move this instantiation to the model, but don't really know what would be optimal, this will work for now.
        for required_key in ["model", "weight_decay", "eps"]:
            assert required_key in kwargs, f"{required_key} required to be in kwargs for AdamW optimizer"
        model = kwargs.pop("model")
        weight_decay = kwargs.pop("weight_decay")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if ( not (any(nd in n for nd in no_decay)) )],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if ( (any(nd in n for nd in no_decay)) )], "weight_decay": 0.0},
        ]
        return AdamW(optimizer_grouped_parameters, **kwargs)
    else:
        raise NotImplementedError

def get_scheduler(custom, **kwargs) -> LRScheduler:
    if custom == "linear_warmup":
        for required_key in ["num_training_steps", "optimizer", "warmup_steps"]:
            assert required_key in kwargs, f"{required_key} required to be in kwargs for linear_warmup scheduler"
        def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
            """ Create a schedule with a learning rate that decreases linearly after
            linearly increasing during a warmup period.
            """

            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(
                    0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
                )

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

        scheduler = get_linear_schedule_with_warmup(
            kwargs["optimizer"], num_warmup_steps=kwargs["warmup_steps"], num_training_steps=kwargs["num_training_steps"]
        )
        return scheduler
    else:
        raise NotImplementedError()


class AwesomeAlignTrainer:
    ''' AwesomeAlignTrainer for replicating awesomealign. Then will want to extend this base class for other purposes and experiments. '''
    def __init__(self, 
                 model: AwesomeAligner, 
                 datasetloaders: AwesomeAlignDatasetLoaders, 
                 logger: BaseLogger, 
                 get_optimizer: Callable[..., Optimizer], 
                 get_scheduler: Callable[..., LRScheduler], 
                 max_grad_norm: float,
                 device: str, 
                 max_steps: int, 
                 max_epochs: float, 
                 log_every_n_steps: int, 
                 val_every_n_steps: int,
                 output_dir: str,
                 ):
        if (max_steps < 0 and max_epochs < 0) or (max_steps >= 0 and max_epochs >= 0):
            raise InterruptedError(f"exactly one of max_steps or max_epochs be non negative, {max_epochs=}, {max_steps=}")
        self.model = model
        self.datasetloaders = datasetloaders
        self.logger = logger
        self.seed = 2
        self.get_optimizer = get_optimizer
        self.get_scheduler = get_scheduler
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.val_every_n_steps = val_every_n_steps
        self.output_dir = output_dir

    def initialize_train(self, config):
        ''' What is the initialize training model in charge of?
        this function should instantiate the model, and datsetloaders, and organize them to be used for eventual training...
        Who should be defining the collate function?
        '''
        self.config = config
        if self.seed > 0:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        # taken from variable location depending on the model. might have to change this in future.
        self.datasetloaders.setup(block_size=512) 
        
        if self.max_steps < 0:
            self.max_steps = int(self.max_epochs * np.ceil(len(self.datasetloaders.train_dataset) / self.datasetloaders.batch_size))
        elif self.max_epochs < 0:
            self.max_epochs = self.max_steps * self.datasetloaders.batch_size / len(self.datasetloaders.train_dataset)
        
        self.logger.watch(self.model)
        
        self.optimizer = self.get_optimizer(model=self.model)
        self.scheduler = self.get_scheduler(optimizer=self.optimizer, num_training_steps=self.max_steps)


        self.current_epoch = 0
        self.current_global_step = 0

    def train_loop(self, progress_bar: tqdm.tqdm, formatted_string_postfix: str):
        self.model.train()
        for batch in self.datasetloaders.train_dataloader():
            # #debug shit:
            # if self.current_global_step < 39035:
            #     self.current_global_step += 1
            #     progress_bar.update()
            #     continue
            batch = batch.to(self.device)
            # torch.autograd.set_detect_anomaly(True)
            # self.model.eval() # this for debugging and to remove randomness of dropout
            losses = self.model.training_step(batch)
            # each loss should be backpropped seperately. how will this work? I can make the awesome align trainer dependant on the model, and the losses...
            # will just leave it for now, and see how bad it is later.
            loss = losses["loss_per_batch"].mean()
            # loss.backward()

            progress_bar.set_postfix_str(formatted_string_postfix.format_map({"loss": loss.detach().item()}))
            progress_bar.update()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.current_global_step % self.log_every_n_steps == 0 or self.current_global_step + 1 >= self.max_steps:
                log_dict = {k: (v.detach().mean().item() if isinstance(v, torch.Tensor) else v) for k, v in losses.items()}
                log_dict.update({
                    "lr": self.scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm.item(),
                })
                if self.device == "cuda":
                    log_dict.update({f"mem_on_{self.device}": torch.cuda.memory_allocated() / (2**30) })
                elif self.device == "mps":
                    log_dict.update({f"mem_on_{self.device}": torch.mps.current_allocated_memory()})
                log_dict = {("train_" + key): value for key, value in log_dict.items()}
                self.logger.log(log_dict, step=self.current_global_step)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            self.current_global_step += 1
            if self.current_global_step % self.val_every_n_steps == 0:
                self.validation_loop()
            if self.current_global_step >= self.max_steps:
                break

    def fit(self, config):
        self.initialize_train(config)
        self.validation_loop()
        formatted_string_postfix = "loss: {loss:3.5f}"
        progress_bar = tqdm.trange(self.max_steps, postfix=formatted_string_postfix.format_map({"loss": 0}))
        while self.current_epoch < self.max_epochs:
            self.train_loop(progress_bar, formatted_string_postfix)
            self.current_epoch += 1
        if self.current_global_step % self.val_every_n_steps != 0:
            # if we haven't already validated after the last step, then we will run validate loop.
            self.validation_loop()
        self.test_loop()
        # self.save_checkpoint()

    def validation_loop(self):
        with torch.no_grad():
            was_training = self.model.training
            self.model.eval()
            log_dict_total = {}
            for dataset_name, val_dataloader in self.datasetloaders.val_dataloaders_iterator():
                progress_bar_val_iterator: Iterable[CollateFnReturn] = tqdm.tqdm(val_dataloader, desc=f"Val_{dataset_name}") 
                metric = AwesomeAlignerValMetrics()
                for batch in progress_bar_val_iterator:
                    batch.to(self.device)
                    metric.update(self.model.validation_step(batch))

                log_dict = metric.compute()
                log_dict = {(f"eval_{dataset_name}_{key}"): val for key, val in log_dict.items()}
                log_dict_total.update(log_dict)
            print(log_dict_total, self.current_global_step)
            self.logger.log(log_dict_total, step=self.current_global_step)
            self.model.train(was_training)

    def test_loop(self):
        # should run alignment for all languages on the test sets.
        pass

    def load_checkpoint(self):
        # save_dict = {"epoch": self.epoch, "step": self.global_step}
        # save_dict.update(self.state)
        # torch.save(save_dict, f"{self.save_dir}/{self.}")
        # should I be saving just the model, or also the config which was used to create it? This is automatically saved with the model due to hydra.
        pass
    def save_checkpoint(self, name):
        torch.save(
            {
                "optimizer": self.optimizer,
                "scheduler": self.scheduler,
                "model": self.model,
                "current_epoch": self.current_epoch,
                "current_global_step": self.current_global_step,
                "config": self.config, # must load in the datasetloaders, and other stuff probably
            },
            os.path.join(self.output_dir, "checkpoints", name)
        ) 
        # TODO: look into saving and loading partial checkpoints: https://github.com/pranav-putta/lm-nav/blob/main/lmnav/ppo_train.py#L560
        # I particularly have to be careful about loading the config into the trainer, and get the right state loaded into the optimizer, and pick up on the same wandb log.

# %%
