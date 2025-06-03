import os
from pathlib import Path

import pydrantic

from capsules.clients.together import TogetherClient
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.kv_initialization.strategies.summarization import KVCacheInitFromSummary
from capsules.train import (
    EvalDatasetConfig,
    GenerateDatasetConfig,
    TrainConfig,
    TrainableCache,
)
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.utils import WandBConfig


config = TrainConfig(
    name=Path(__file__).stem,
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    ),
    dataset=CapsuleDataset.Config(
        data_sources=[
            ("hazy-research/capsules/m03d13_loogle_next_section_memorize:v0", None),
            ("hazy-research/capsules/m03d13_loogle_next_section_memorize:v1", None),
            ("hazy-research/capsules/m03d13_loogle_previous_section_memorize:v0", None),
            ("hazy-research/capsules/m03d12_basic_qa_loogle_doc:v4", None),
        ],
        is_wandb=True,
        label_type="tokens",
    ),
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="gpt4o_completions",
            local_batch_size=16,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m03d12_gold_q_loogle_doc_openai:v2", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        )
    ],
    generate_datasets=[
        GenerateDatasetConfig(
            name_for_wandb="gpt4o_completions",
            dataset=CapsuleGenerateDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m03d12_gold_q_loogle_doc_openai:v2", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        )
    ],
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=1024,
    ),
    epochs=2,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    train_batch_size=2,
    eval_every_n_steps=128,
    # generate_every_n_steps=128,
    generate_max_new_tokens=64,
    loss_type="tokens",
)

if __name__ == "__main__":
    pydrantic.main([config])
# type: ignore
