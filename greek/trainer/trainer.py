from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class Trainer:
    ''' BaseTrainer '''
    def __init__(self, model: nn.Module, ):
        pass

    def initialize_train(self):
        pass

    def train(self):
        pass
