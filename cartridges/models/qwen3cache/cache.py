import torch 
import torch.nn as nn
import torch.nn.functional as F

class TrainableKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, input_ids, seq_ids, position_ids):
        return input_ids, seq_ids, position_ids