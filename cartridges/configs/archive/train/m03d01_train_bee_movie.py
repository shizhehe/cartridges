import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.config import HFModelConfig
from capsules.train import TrainConfig, TrainableCache, CapsuleDataset
from capsules.tasks.finance.dataset import FinanceChunkDataset
from capsules.utils import WandBConfig


file_name = Path(__file__).stem


config = TrainConfig(
    name=file_name,
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    ),
    dataset=CapsuleDataset.Config(
        # TODO(rse): update the path to point to the output of the generate run
        data_path="...",  # path output by the generate run
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
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
)

if __name__ == "__main__":
    # launch the config
    pydrantic.main([config])

