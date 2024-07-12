import torch.nn as nn
import torch

class EncoderAligner(nn.Module):
    def __init__(self, encoder: nn.Module, layer_number: int, ):
        super().__init__()