import os

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.train import TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.utils import WandBConfig


configs = []
# for lr in [5e-3, 5e-4, 1e-3, 1e-2, 5e-2]:
for num_tokens in [
    1024
    # 128, 256, 512, 1024, 2048, 4096, 8192, 16384
]:
    config = TrainConfig(
        name=f"sweep_num_tokens_{num_tokens}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        dataset=CapsuleDataset.Config(
            data_sources=[
                # ("hazy-research/capsules/m03d09_between_sections_memorization:v0", None),
                # ("hazy-research/capsules/m03d09_next_section_memorization:v0", None),
                # ("hazy-research/capsules/m03d09_previous_section_memorization:v0", None),
                ("hazy-research/capsules/m03d07_basic_qa_train:v4", None),
            ],
            is_wandb=True,
            label_type="tokens",
        ),
        eval_every_n_steps=128,
        eval_dataset=CapsuleDataset.Config(
            data_sources=[
                ("hazy-research/capsules/m03d07_basic_qa_train:v5", 128)
                # ("hazy-research/capsules/m03d10_basic_qa_test_openai:v0", 256)
            ],
            is_wandb=True,
            label_type="tokens",
        ),
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens,
        ),
        generate_every_n_steps=128,
        generate_dataset=CapsuleGenerateDataset.Config(
            data_sources=[("hazy-research/capsules/m03d10_basic_qa_test_openai:v0", 16)],
            is_wandb=True,
            label_type="tokens",
        ),
        loss_type="tokens",
        save_every_n_steps=512,
        epochs=1,
        lr=5e-3,
        wandb=WandBConfig(
            project="capsules",
            tags=["cache_tuning", "development"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        batch_size=4,
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
