import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig, PeftConfig
from capsules.datasets import CapsuleDataset
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
# Create configurations with different LoRA ranks to test parameter efficiency
RANKS = [ int(x) for x in os.environ["CAPSULES_LORA_RANKS"].split(",") ]

for lora_rank in RANKS:
    config = TrainConfig(
        name=f"{file_name}_rank={lora_rank}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            tuning_method="peft",  # Use PEFT instead of custom prefix tuning
            peft=PeftConfig(
                enabled=True,  # Enable PEFT
                method="lora",  # Use LoRA
                r=lora_rank,  # LoRA rank
                alpha=2 * lora_rank,  # LoRA scaling factor (typically 2*rank)
                dropout=0.05,  # LoRA dropout
                # Updated target modules for LLaMA 3 architecture
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
        ),
        dataset=CapsuleDataset.Config(
            data_sources=[
                # Using the full context answers dataset
                ("hazy-research/capsules/m03d18_generate_longhealth_p01_full_context:v0", None),
            ],  
            is_wandb=True,
            label_type="tokens",
        ),
        generate_every_n_steps=16,
        generate_datasets=[
            GenerateDatasetConfig(
                name_for_wandb="multiple_choice_generations",
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=["patient_01"],
                    max_questions=128,
                    cot=True
                )
            ),
        ],
        eval_every_n_steps=16,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb="longhealth_multiple_choice",
                local_batch_size=16,
                dataset=LongHealthEvalDataset.Config(
                    patient_ids=["patient_01"],
                    max_questions=256,
                    label_type="tokens",
                    data_sources=[]  # ignore this arg
                )
            ),
            EvalDatasetConfig(
                name_for_wandb="generated_questions",
                local_batch_size=16,
                dataset=CapsuleDataset.Config(
                    data_sources=[
                        ("hazy-research/capsules/m03d12_longhealth_p01_basic_qa_test:v0", None),
                    ],
                    is_wandb=True,
                    label_type="tokens",
                ),
            )
        ],
        generate_max_new_tokens=512,
        # No need for kv_cache_initializer when using PEFT, but keeping for consistency 
        # with original implementation (it won't be used with tuning_method="peft")
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=512
        ),
        loss_type="tokens",
        save_every_n_steps=64,
        epochs=1,
        lr=5e-3,
        wandb=WandBConfig(
            project="capsules",
            tags=["peft", "lora", "longhealth", "development"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        train_batch_size=6,
    )
    configs.append(config)

# Add a prefix tuning configuration for comparison
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
            # Set task_type for clarity
            task_type="CAUSAL_LM",
        )
    ),
    dataset=CapsuleDataset.Config(
        data_sources=[
            # Using the full context answers dataset
            ("hazy-research/capsules/m03d18_generate_longhealth_p01_full_context:v0", None),
        ],  
        is_wandb=True,
        label_type="tokens",
    ),
    generate_every_n_steps=16,
    generate_datasets=[
        GenerateDatasetConfig(
            name_for_wandb="multiple_choice_generations",
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=["patient_01"],
                max_questions=128,
                cot=True
            )
        ),
    ],
    eval_every_n_steps=16,
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="longhealth_multiple_choice",
            local_batch_size=16,
            dataset=LongHealthEvalDataset.Config(
                patient_ids=["patient_01"],
                max_questions=256,
                label_type="tokens",
                data_sources=[]  # ignore this arg
            )
        ),
        EvalDatasetConfig(
            name_for_wandb="generated_questions",
            local_batch_size=16,
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m03d12_longhealth_p01_basic_qa_test:v0", None),
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        )
    ],
    generate_max_new_tokens=512,
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=512
    ),
    loss_type="tokens",
    save_every_n_steps=64,
    epochs=1,
    lr=5e-3,
    wandb=WandBConfig(
        project="capsules",
        tags=["peft", "prefix_tuning", "longhealth", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    train_batch_size=6,
)
configs.append(config_prefix)

if __name__ == "__main__":
    pydrantic.main(configs)