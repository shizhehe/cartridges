import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization.random import KVFromRandomText
from cartridges.train import PerplexityEvalConfig, GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeTrainDataset
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem
bs = 64

data_sources = [
    "/home/sabri/code-memory/outputs/2025-07-06-13-37-13-gmail_synthesis/gmail_synthesis-0/artifact/dataset.pkl"
]

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
    ),
    kv_cache_initializer=KVFromRandomText.Config(
        max_tokens=2048
    ),
    
    lr=2e-2,
    loss_type="logits",
    epochs=2,
    global_batch_size=bs,
    local_batch_size=4,
    use_batch_sampler=True,

    dataset=CartridgeTrainDataset.Config(
        data_sources=[
            (source, None)
            for source in data_sources
        ],
        max_sequence_length=1024,
        is_wandb=True,
        label_type="logits",
        top_k_logits=20,
    ),

    
    save_every_n_steps=512,
    generate_every_n_steps=512,
    generate_max_new_tokens=512,
    generate_evals=[],
    eval_every_n_steps=256,
    eval_datasets=[],
    distributed_backend="gloo",

    wandb=WandBConfig(
        project="cartridges",
        tags=["train", "gmail"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"{file_name}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
)


if __name__ == "__main__":
    pydrantic.main(config)