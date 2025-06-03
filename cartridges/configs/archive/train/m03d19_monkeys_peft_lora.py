import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.train import TrainConfig, GenerateDatasetConfig, EvalDatasetConfig
from capsules.config import HFModelConfig, PeftConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem
configs = []

# LoRA configuration for monkeys dataset
config = TrainConfig(
    name=f"{file_name}",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        tuning_method="peft",  # Use PEFT instead of custom prefix tuning
        peft=PeftConfig(
            enabled=True,  # Enable PEFT
            method="lora",  # Use LoRA
            r=16,  # LoRA rank
            alpha=32,  # LoRA scaling factor
            dropout=0.05,  # LoRA dropout
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target modules for LoRA
        )
    ),
    dataset=CapsuleDataset.Config(
        data_sources=[
            ("hazy-research/capsules/monkeys_next_section_pred:v0", None),
            ("hazy-research/capsules/m03d17_monkeys_basic_qa_logits:v0", None),
        ],
        is_wandb=True,
        label_type="logits",  # Matching the label type from monkeys training config
    ),
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="gpt4o_completions",
            batch_size=16,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m03d17_basic_qa_monkeys_openai:v0", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
        EvalDatasetConfig(
            name_for_wandb="mmlu",
            batch_size=16,
            dataset=MMLUEvalDataset.Config(num_samples=256),
        ),
    ],
    generate_datasets=[
        GenerateDatasetConfig(
            name_for_wandb="gpt4o_completions",
            dataset=CapsuleGenerateDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m03d17_basic_qa_monkeys_openai:v0", 16)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        )
    ],
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=1000,
        num_frozen_tokens=1,
    ),
    loss_type="logits",  # Matching the loss type from monkeys training config
    save_every_n_steps=128,
    epochs=1,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["peft", "lora", "monkeys", "development"],
        entity="hazy-research",
    ),
    train_batch_size=2,
    eval_every_n_steps=128,
    generate_every_n_steps=128,
    generate_max_new_tokens=64,
)
configs.append(config)

# Prefix tuning configuration for monkeys dataset
config_prefix = TrainConfig(
    name=f"{file_name}_prefix_tuning",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        tuning_method="peft",
        peft=PeftConfig(
            enabled=True,
            method="prefix_tuning",
            num_virtual_tokens=30,
            encoder_hidden_size=128, 
            prefix_projection=True,
        )
    ),
    dataset=CapsuleDataset.Config(
        data_sources=[
            ("hazy-research/capsules/monkeys_next_section_pred:v0", None),
            ("hazy-research/capsules/m03d17_monkeys_basic_qa_logits:v0", None),
        ],
        is_wandb=True,
        label_type="logits",
    ),
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="gpt4o_completions",
            batch_size=16,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m03d17_basic_qa_monkeys_openai:v0", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),
        EvalDatasetConfig(
            name_for_wandb="mmlu",
            batch_size=16,
            dataset=MMLUEvalDataset.Config(num_samples=256),
        ),
    ],
    generate_datasets=[
        GenerateDatasetConfig(
            name_for_wandb="gpt4o_completions",
            dataset=CapsuleGenerateDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m03d17_basic_qa_monkeys_openai:v0", 16)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        )
    ],
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=1000,
        num_frozen_tokens=1,
    ),
    loss_type="logits",
    save_every_n_steps=128,
    epochs=1,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["peft", "prefix_tuning", "monkeys", "development"],
        entity="hazy-research",
    ),
    train_batch_size=2,
    eval_every_n_steps=128,
    generate_every_n_steps=128,
    generate_max_new_tokens=64,
)
configs.append(config_prefix)

# Add output_dir if environment variable is set
if "CAPSULES_OUTPUT_DIR" in os.environ:
    for cfg in configs:
        cfg.output_dir = os.environ["CAPSULES_OUTPUT_DIR"]

if __name__ == "__main__":
    pydrantic.main(configs)