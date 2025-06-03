import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.config import HFModelConfig
from capsules.train import TrainConfig, TrainableCache
from capsules.tasks.finance.dataset import FinanceChunkDataset
from capsules.utils import WandBConfig


file_name = Path(__file__).stem


config = TrainConfig(
    name=FormatStringVariable(f"{file_name}_cs{{dataset.chunk_size}}_nt{{cache.num_tokens}}_lr{{lr}}"), #_l{model.pretrained_model_name_or_path}"),
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct"
    ),
    dataset=FinanceChunkDataset.Config(
        chunk_size=4096,
        row_idx=0,
    ),
    cache=TrainableCache.Config(
        num_tokens=2048,
    ),
    epochs=128,
    cache_init="document",
    lr=5e-3, 
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
    ),
    output_dir=os.environ["capsules_OUTPUT_DIR"],
)

if __name__ == "__main__":
    # launch the config
    pydrantic.main([config])
