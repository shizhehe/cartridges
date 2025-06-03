import os
from pathlib import Path

import pydrantic


from cartridges.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.tasks.mmlu import MMLUEvalDataset
from cartridges.train import (
    EvalDatasetConfig,
    GenerateDatasetConfig,
    TrainConfig,
)
from cartridges.models.config import HFModelConfig, PeftConfig
from cartridges.datasets import CartridgeDataset, CartridgeGenerateDataset
from cartridges.utils import WandBConfig


MONKEYS_TITLE = (
    "Large Language Monkeys: Scaling Inference Compute with Repeated Sampling"
)
MINIONS_TITLE = (
    "Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models"
)

EVAL_BS = 16
TRAIN_BS = 4

lora_rank = 32

configs = []
for lr in [5e-3, 5e-4, 5e-5]:
    for lora_rank in [32, 64, 128, 256]:
        for global_bs in [32, 64, 128]:
            configs.append(TrainConfig(
                name=f"{Path(__file__).stem}_rank{lora_rank}_lr{lr:.0e}_bs{global_bs}",
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
                dataset=CartridgeDataset.Config(
                    data_sources=[
                        ("hazy-research/Cartridges/monkeys_teaching_data_chunks:v0", None),
                    ],
                    is_wandb=True,
                    label_type="logits",
                ),
                eval_datasets=[
                    EvalDatasetConfig(
                        name_for_wandb="monkeys_chunks",
                        local_batch_size=EVAL_BS,
                        dataset=CartridgeDataset.Config(
                            data_sources=[
                                ("hazy-research/Cartridges/openai_eval_monkeys_chunks:v0", None)
                            ],
                            is_wandb=True,
                            label_type="tokens",
                        ),
                    ),
                    EvalDatasetConfig(
                        name_for_wandb="monkeys_whole",
                        local_batch_size=EVAL_BS,
                        dataset=CartridgeDataset.Config(
                            data_sources=[
                                ("hazy-research/Cartridges/openai_eval_monkeys_whole_doc:v0", None)
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
                        dataset=CartridgeGenerateDataset.Config(
                            data_sources=[
                                ("hazy-research/Cartridges/openai_eval_monkeys_chunks:v0", 16)
                            ],
                            is_wandb=True,
                            label_type="tokens",
                        ),
                    ),
                    GenerateDatasetConfig(
                        name_for_wandb="monkeys_whole",
                        dataset=CartridgeGenerateDataset.Config(
                            data_sources=[
                                (
                                    "hazy-research/Cartridges/eval_openai_questions_3b_answers_monkeys_whole_doc:v0",
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
                    project="cartridges",
                    tags=["cache_tuning", "development"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
                local_batch_size=TRAIN_BS,
                global_batch_size=global_bs,
                eval_every_n_steps=128,
                generate_every_n_steps=256,
                generate_max_new_tokens=64,
                save_every_n_steps=128,
                loss_type="logits",
            ))

if __name__ == "__main__":
    pydrantic.main(configs)
