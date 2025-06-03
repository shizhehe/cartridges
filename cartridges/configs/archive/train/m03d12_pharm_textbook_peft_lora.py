import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.train import TrainConfig, GenerateDatasetConfig
from capsules.config import HFModelConfig, PeftConfig
from capsules.datasets import CapsuleDataset
from capsules.tasks.openstax.datasets import OpenStaxMultipleChoiceGenerateDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem
configs = []
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
            ("hazy-research/capsules/m03d11_basic_qa_train_pharm_textbook:v0", None),
        ],
        is_wandb=True,
        label_type="tokens",
    ),
    generate_every_n_steps=512,
    generate_datasets=[
        GenerateDatasetConfig(
            dataset=OpenStaxMultipleChoiceGenerateDataset.Config(
                path=str(Path(__file__).parent.parent.parent.parent.parent / "data/tasks/openstax/pharm_questions.json")
            ),
            name_for_wandb="generate_pharm_qa",
        )
    ],
    generate_max_new_tokens=10,
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=5_000
    ),
    loss_type="tokens",
    save_every_n_steps=512,
    epochs=1,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["peft", "lora", "development"],
        entity="hazy-research",
    ),
    train_batch_size=2,
)
configs.append(config)

# Another example with prefix tuning
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
            ("hazy-research/capsules/m03d11_basic_qa_train_pharm_textbook:v0", None),
        ],
        is_wandb=True,
        label_type="tokens",
    ),
    generate_every_n_steps=512,
    generate_datasets=[
        GenerateDatasetConfig(
            dataset=OpenStaxMultipleChoiceGenerateDataset.Config(
                path=str(Path(__file__).parent.parent.parent.parent.parent / "data/tasks/openstax/pharm_questions.json")
            ),
            name_for_wandb="generate_pharm_qa",
        )
    ],
    generate_max_new_tokens=10,
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=5_000
    ),
    loss_type="tokens",
    save_every_n_steps=512,
    epochs=1,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["peft", "prefix_tuning", "development"],
        entity="hazy-research",
    ),
    train_batch_size=2,
)
configs.append(config_prefix)

if __name__ == "__main__":
    pydrantic.main(configs)