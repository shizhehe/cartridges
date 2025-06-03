import os
from pathlib import Path

import pydrantic



# from capsules.data.paths import AZALIA_DPO_TITLE, AZALIA_FAST_TITLE
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import (
    EvalDatasetConfig,
    GenerateDatasetConfig,
    TrainConfig,
)
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.utils import WandBConfig

AZALIA_DPO_TITLE = "Device Placement Optimization with Reinforcement Learning"
AZALIA_FAST_TITLE = "A Full-Stack Search Technique for Domain Optimized Deep Learning Accelerators"


config = TrainConfig(
    name=Path(__file__).stem,
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    ),
    dataset=CapsuleDataset.Config(
        data_sources=[
            ("hazy-research/capsules/generate_azalia_fast:v0", None),
        ],
        is_wandb=True,
        label_type="logits",
    ),
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="fast_whole_doc",
            local_batch_size=16,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/azalia_fast_whole_doc_openai_3b_eval:v0", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
        EvalDatasetConfig(
            name_for_wandb="fast_chunks",
            local_batch_size=8,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/azalia_fast_chunks_openai_3b_eval:v0", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
        EvalDatasetConfig(
            name_for_wandb="mmlu",
            local_batch_size=8,
            dataset=MMLUEvalDataset.Config(num_samples=256),
        ),
    ],
    generate_datasets=[
        GenerateDatasetConfig(
            name_for_wandb="fast_chunks",
            dataset=CapsuleGenerateDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/azalia_fast_chunks_openai_3b_eval:v0", 16)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
        GenerateDatasetConfig(
            name_for_wandb="fast_whole_doc",
            dataset=CapsuleGenerateDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/azalia_fast_whole_doc_openai_3b_eval:v0", 16)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
    ],
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=None,
    ),
    epochs=0,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    local_batch_size=2,
    global_batch_size=2*7,
    eval_every_n_steps=128,
    generate_every_n_steps=256,
    generate_max_new_tokens=64,
    save_every_n_steps=128,
    loss_type="logits",
)

if __name__ == "__main__":
    pydrantic.main([config])
