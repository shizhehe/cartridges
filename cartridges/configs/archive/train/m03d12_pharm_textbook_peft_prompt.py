import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.train import TrainConfig
from capsules.config import HFModelConfig, PeftConfig
from capsules.datasets import CapsuleDataset
from capsules.tasks.openstax.datasets import OpenStaxMultipleChoiceGenerateDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem
configs = []

# Example with prompt tuning
config_prompt = TrainConfig(
    name=f"{file_name}",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        tuning_method="peft",
        peft=PeftConfig(
            enabled=True,
            method="prompt_tuning",
            num_virtual_tokens=20,
            prompt_tuning_init="TEXT",
            prompt_tuning_init_text="Answer the following question about pharmacology: ",
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
        TrainConfig.GenerateDatasetConfig(
            dataset=OpenStaxMultipleChoiceGenerateDataset.Config(
                path=str(Path(__file__).parent.parent.parent.parent / "data/tasks/openstax/pharm_questions.json")
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
        tags=["peft", "prompt_tuning", "development"],
        entity="hazy-research",
    ),
    train_batch_size=2,
)
configs.append(config_prompt)

# Example with p-tuning
config_p_tuning = TrainConfig(
    name=f"{file_name}_p_tuning",
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        tuning_method="peft",
        peft=PeftConfig(
            enabled=True,
            method="p_tuning",
            num_virtual_tokens=20,
            encoder_hidden_size=512,
            encoder_reparameterization_type="MLP",
            encoder_dropout=0.1,
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
        TrainConfig.GenerateDatasetConfig(
            dataset=OpenStaxMultipleChoiceGenerateDataset.Config(
                path=str(Path(__file__).parent.parent.parent.parent / "data/tasks/openstax/pharm_questions.json")
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
        tags=["peft", "p_tuning", "development"],
        entity="hazy-research",
    ),
    train_batch_size=2,
)
configs.append(config_p_tuning)

if __name__ == "__main__":
    pydrantic.main(configs)