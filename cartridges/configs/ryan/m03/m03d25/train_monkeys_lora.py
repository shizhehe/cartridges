import os
from pathlib import Path

import pydrantic


from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import (
    EvalDatasetConfig,
    GenerateDatasetConfig,
    TrainConfig,
)
from capsules.config import HFModelConfig, PeftConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.utils import WandBConfig


MONKEYS_TITLE = (
    "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"
)
MINIONS_TITLE = (
    "Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models"
)

EVAL_BS = 16
TRAIN_BS = 2

# EVAL_BS = 16
# TRAIN_BS = 2

lora_rank = 32

config = TrainConfig(
    name=f"{Path(__file__).stem}_rank_{lora_rank}",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        tuning_method="peft",  # Use PEFT instead of custom prefix tuning
        peft=PeftConfig(
            enabled=True,  # Enable PEFT
            method="lora",  # Use LoRA
            r=lora_rank,  # LoRA rank
            alpha=int(2 * lora_rank),  # LoRA scaling factor (typically 2*rank)
            dropout=0.05,  # LoRA dropout
            # Updated target modules for LLaMA 3 architecture
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    ),
    dataset=CapsuleDataset.Config(
        data_sources=[
            ("hazy-research/capsules/monkeys_teaching_data_chunks:v0", None),
        ],
        is_wandb=True,
        label_type="logits",
    ),
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="monkeys_chunks",
            local_batch_size=EVAL_BS,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/openai_eval_monkeys_chunks:v0", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
        EvalDatasetConfig(
            name_for_wandb="monkeys_whole",
            local_batch_size=EVAL_BS,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/openai_eval_monkeys_whole_doc:v0", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
        EvalDatasetConfig(
            name_for_wandb="mmlu",
            local_batch_size=16,
            dataset=MMLUEvalDataset.Config(num_samples=256),
        ),
    ],
    generate_datasets=[
        GenerateDatasetConfig(
            name_for_wandb="monkeys_chunk",
            dataset=CapsuleGenerateDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/openai_eval_monkeys_chunks:v0", 16)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
        GenerateDatasetConfig(
            name_for_wandb="monkeys_whole",
            dataset=CapsuleGenerateDataset.Config(
                data_sources=[
                    (
                        "hazy-research/capsules/eval_openai_questions_3b_answers_monkeys_whole_doc:v0",
                        16,
                    )
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
    ],
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=1000,
        num_frozen_tokens=1,
    ),
    epochs=1,
    lr=5e-4,
    wandb=WandBConfig(
        project="capsules",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    local_batch_size=TRAIN_BS,
    global_batch_size=32 * TRAIN_BS,
    eval_every_n_steps=128,
    generate_every_n_steps=256,
    generate_max_new_tokens=64,
    save_every_n_steps=128,
    loss_type="logits",
)

if __name__ == "__main__":
    pydrantic.main([config])
