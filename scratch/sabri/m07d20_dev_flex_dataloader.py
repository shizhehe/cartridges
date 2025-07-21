
import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.initialization.random import KVFromRandomText
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeTrainDataset
from cartridges.utils import WandBConfig

data_sources = [
    "/home/sabri/cartridges/outputs/2025-07-13-09-04-32-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_p10_n65536-0/artifact/dataset.pkl"
]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")

config =CartridgeTrainDataset.Config(
    data_sources=[
        (source, None)
        for source in data_sources
    ],
    max_sequence_length=1024,
    is_wandb=True,
    label_type="logits",
    top_k_logits=20,
)
dataset = config.instantiate(tokenizer=tokenizer)


dataloader = DataLoader(
    dataset,
    collate_fn=dataset.collate,
    batch_size=4
)

for batch in dataloader:
    breakpoint()





