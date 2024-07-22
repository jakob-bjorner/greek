#%%
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from greek.model.model import AwesomeAligner, CollateFnReturn
from greek.logger.logger import BaseLogger
from greek.datasetloaders.datasetloaders import AwesomeAlignDatasetLoaders
from typing import Any, Callable, Iterable
import random
import numpy as np
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler
import tqdm

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
                 device: str, 
                 max_steps: int, 
                 max_epochs: float, 
                 log_every_n_steps: int, 
                 val_every_n_steps: int
                 ):
        if (max_steps <= 0 and max_epochs <= 0) or (max_steps > 0 and max_epochs > 0):
            raise InterruptedError(f"exactly one of max_steps or max_epochs be positive, {max_epochs=}, {max_steps=}")
        self.model = model
        self.datasetloaders = datasetloaders
        self.logger = logger
        self.seed = 1
        self.get_optimizer = get_optimizer
        self.get_scheduler = get_scheduler
        self.device = device
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.val_every_n_steps = val_every_n_steps

    def initialize_train(self):
        ''' What is the initialize training model in charge of?
        this function should instantiate the model, and datsetloaders, and organize them to be used for eventual training...
        Who should be defining the collate function?
        '''
        if self.seed > 0:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        # taken from variable location depending on the model. might have to change this in future.
        self.datasetloaders.setup(block_size=512) 
        
        if self.max_steps < 0:
            self.max_steps = self.max_epochs * len(self.datasetloaders.train_dataset) / self.datasetloaders.batch_size
        elif self.max_epochs < 0:
            self.max_epochs = self.max_steps * self.datasetloaders.batch_size / len(self.datasetloaders.train_dataset)
        
        self.optimizer = self.get_optimizer(model=self.model)
        self.scheduler = self.get_scheduler(optimizer=self.optimizer, num_training_steps=self.max_steps)

        self.current_epoch = 0
        self.current_global_step = 0

    def train_loop(self):
        self.model.train()
        formatted_string_postfix = "loss: {loss:3.5f}"
        progress_bar_iterator: Iterable[CollateFnReturn]  = tqdm.tqdm(self.datasetloaders.train_dataloader(), postfix=formatted_string_postfix.format_map({"loss": 0}))
        for batch in progress_bar_iterator:
            batch = batch.to(self.device)
            losses = self.model.training_step(batch)
            # each loss should be backpropped seperately. how will this work? I can make the awesome align trainer dependant on the model, and the losses...
            # will just leave it for now, and see how bad it is later.
            # TODO: check if backpropping only the one loss is much slower.
            loss = losses["loss"]
            loss.backward()
            if self.current_global_step % self.log_every_n_steps == 0:
                log_dict = {k: v.detach().mean().item() for k, v in losses.items()}
                if self.device == "cuda":
                    log_dict.update({f"mem_on_{self.device}": torch.cuda.memory_allocated()  })
                elif self.device == "mps":
                    log_dict.update({f"mem_on_{self.device}": torch.mps.current_allocated_memory()})
                
                self.logger.log(log_dict, step=self.current_global_step)
            progress_bar_iterator.set_postfix_str(formatted_string_postfix.format_map({"loss": loss.detach().item()}))
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            if self.current_global_step % self.val_every_n_steps == 0:
                self.validation_loop()
            self.current_global_step += 1
            if self.current_global_step >= self.max_steps:
                break

    def fit(self):
        self.initialize_train()
        while self.current_epoch < self.max_epochs:
            self.train_loop()
            self.current_epoch += 1
        self.validation_loop()
        self.test_loop()
        self.save()

    def validation_loop(self):
        was_training = self.model.training
        self.model.eval()
        was_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        # should run alignment for all languages on the val set
        progress_bar_val_iterator: Iterable[CollateFnReturn] = tqdm.tqdm(self.datasetloaders.val_dataloader()) 
        loss_avg = 0
        for batch in progress_bar_val_iterator:
            batch.to(self.device)
            metrics = self.model.validation_step(batch)
            # will want to take the average of all the metrics? alignment error rate, precision, recall do they average well the answer is no, so they will have to be handed appropriately.
            # likely by giving something in metrics which can be accumulated. The trainer becomes much more dependant on the metrics
            loss_avg += metrics["loss"].detach().item() * batch.examples_src.size(0)
        import ipdb
        ipdb.set_trace()

        loss_avg /= len(self.datasetloaders.val_dataset)
        self.logger.log({"loss": loss_avg}, step=self.current_global_step)
        
        if was_training:
            self.model.train()
        torch.set_grad_enabled(was_grad_enabled)
        

    def test_loop(self):
        # should run alignment for all languages on the test sets.
        pass

    def save(self):
        # save_dict = {"epoch": self.epoch, "step": self.global_step}
        # save_dict.update(self.state)
        # torch.save(save_dict, f"{self.save_dir}/{self.}")
        # should I be saving just the model, or also the config which was used to create it? This is automatically saved with the model due to hydra.
        pass

# %%
