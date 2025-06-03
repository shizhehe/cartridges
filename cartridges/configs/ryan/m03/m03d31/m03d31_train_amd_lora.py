import os
from pathlib import Path

import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.tasks.finance import (
    FinanceBenchEvalDataset,
    FinanceBenchGenerateDataset,
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

configs = []
if True:
    generate_evals, ppl_evals = get_evals(
        FinanceEvals,
        DOC_NAME,
        num_samples=16,  # RE: it's silly we have to specify this
        version_tag="v1",
        batch_size=16,
    )

    lora_rank = 384 // 2
    configs.append(
        TrainConfig(
            name=f"{file_name}_{DOC_NAME}_r{lora_rank}",
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
                        "hazy-research/capsules/m03d27_gen_simple_data_amd_2022_10k:v0",
                        None,
                    ),
                ],
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),
            # generate_every_n_steps=128,
            # generate_datasets=[
            #     GenerateDatasetConfig(
            #         name_for_wandb="finance-bench-gt",
            #         dataset=FinanceBenchGenerateDataset.Config(
            #             doc_names=[DOC_NAME],
            #         ),
            #     ),
            #     *generate_evals,
            # ],
            eval_every_n_steps=32,
            eval_datasets=[
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
                    name_for_wandb="mmlu",
                    local_batch_size=16,
                    dataset=MMLUEvalDataset.Config(num_samples=128),
                ),
                *ppl_evals,
            ],
            generate_max_new_tokens=512,
            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=100_000  # unused
            ),
            loss_type="logits",
            save_every_n_steps=128,
            epochs=1,
            lr=5e-4,
            wandb=WandBConfig(
                project="capsules",
                tags=["cache_tuning", "development"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=8,
        )
    )

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)
