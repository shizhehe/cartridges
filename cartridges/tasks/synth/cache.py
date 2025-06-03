
from pydrantic import BaseConfig
import torch.nn as nn

class BaseKVCache(nn.Module):

    class Config(BaseConfig):
        num_tokens: int

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def forward(self, input_ids, position_ids):
        pass