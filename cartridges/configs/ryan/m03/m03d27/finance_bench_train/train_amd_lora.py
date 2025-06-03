import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.tasks.finance import (
    FinanceBenchEvalDataset,
    FinanceBenchGenerateDataset,
    L3bFinanceBenchEvalDataset,
    L3bFinanceBenchGenerateDataset,
)
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig, PeftConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset

from capsules.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
DOC_NAME = "AMD_2022_10K"
bs = 64

lora_rank = 384 // 2

configs = []
if True:
    configs.append(
        TrainConfig(
            name=f"{file_name}_{DOC_NAME}_rank{lora_rank}_bs_{bs}_bugfix",
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
                    (
                        "hazy-research/capsules/m03d27_gen_simple_data_amd_2022_10k_70b:v0",
                        None,
                    ),
                ],
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),
            generate_every_n_steps=64,
            generate_datasets=[
                GenerateDatasetConfig(
                    name_for_wandb="finance-bench",
                    dataset=L3bFinanceBenchGenerateDataset.Config(
                        doc_names=[DOC_NAME],
                    ),
                ),
                GenerateDatasetConfig(
                    name_for_wandb="finance-bench-gt",
                    dataset=FinanceBenchGenerateDataset.Config(
                        doc_names=[DOC_NAME],
                    ),
                ),
                GenerateDatasetConfig(
                    name_for_wandb="anthropic-questions-creative",
                    dataset=CapsuleGenerateDataset.Config(
                        data_sources=[
                            (
                                "hazy-research/capsules/anthropic_questions_higher_temp:v1",
                                16,
                            )
                        ],
                        is_wandb=True,
                        label_type="tokens",
                    ),
                ),
                GenerateDatasetConfig(
                    name_for_wandb="anthropic-questions-factual",
                    dataset=CapsuleGenerateDataset.Config(
                        data_sources=[
                            (
                                "hazy-research/capsules/anthropic_questions_factual:v0",
                                16,
                            )
                        ],
                        is_wandb=True,
                        label_type="tokens",
                    ),
                )
            ],
            eval_every_n_steps=32,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb="finance-ppl",
                    local_batch_size=16,
                    dataset=L3bFinanceBenchEvalDataset.Config(
                        doc_names=[DOC_NAME],
                        cot=False,
                        label_type="tokens",
                        data_sources=[],  # ignore this arg
                    ),
                    only_eval_rank_0=True,
                ),
                EvalDatasetConfig(
                    name_for_wandb="finance-ppl-gt",
                    local_batch_size=16,
                    dataset=FinanceBenchEvalDataset.Config(
                        doc_names=[DOC_NAME],
                        cot=False,
                        label_type="tokens",
                        data_sources=[],  # ignore this arg
                    ),
                    only_eval_rank_0=True,
                ),
                EvalDatasetConfig(
                    name_for_wandb="anthropic-questions",
                    local_batch_size=16,
                    dataset=CapsuleDataset.Config(
                        data_sources=[
                            (
                                "hazy-research/capsules/m03d27_amd10k_sonnet37_manual_eval:v1",
                                None,
                            )
                        ],
                        is_wandb=True,
                        label_type="tokens",
                    ),
                ),
                
                EvalDatasetConfig(
                    name_for_wandb="anthropic-questions-creative",
                    local_batch_size=16,
                    dataset=CapsuleDataset.Config(
                        data_sources=[
                            ("hazy-research/capsules/anthropic_questions_higher_temp:v1", None)
                        ],
                        is_wandb=True,
                        label_type="tokens",
                    ),
                    only_eval_rank_0=True,
                ),

                EvalDatasetConfig(
                    name_for_wandb="anthropic-questions-factual",
                    local_batch_size=16,
                    dataset=CapsuleDataset.Config(
                        data_sources=[
                            ("hazy-research/capsules/anthropic_questions_factual:v0", None)
                        ],
                        is_wandb=True,
                        label_type="tokens",
                    ),
                ),

                EvalDatasetConfig(
                    name_for_wandb="mmlu",
                    local_batch_size=16,
                    dataset=MMLUEvalDataset.Config(num_samples=128),
                ),
            ],
            generate_max_new_tokens=1024,
            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=100_000, # unused
            ),
            loss_type="logits",
            save_every_n_steps=64,
            epochs=1,
            lr=5e-4,
            wandb=WandBConfig(
                project="capsules",
                tags=["cache_tuning", "development"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=1,
        )
    )

if __name__ == "__main__":
    pydrantic.main(configs)
