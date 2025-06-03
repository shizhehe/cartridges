import os
from pathlib import Path

import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.optim import CosWithWarmup
from capsules.tasks.finance import (
    FinanceBenchEvalDataset,
)
from capsules.tasks.mmlu import MMLUEvalDataset, MMLUGenerateDataset, mmlu_subset
from capsules.tasks.mtob import mtob_eval_datasets, mtob_generate_datasets
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import (
    CapsuleDatasetLatest,
)

from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64

configs = []
if True:

    # for num_tokens, lr in [(6144, 0.01), (6144, 0.02,), (6144, 0.04), (6144, 0.005), (6144, 0.1)]:
    for num_tokens in [1024]:
        for lr in [0.005]:
            sources = [
                (
                    "hazy-research/capsules/m05d06_generate_monkeys_latex_summarization_question_use_case_creative_12288:v0",
                    None,
                ),
            ]
            configs.append(
                TrainConfig(
                    name=f"{file_name}_nt{num_tokens}",
                    model=HFModelConfig(
                        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                    ),
                    dataset=CapsuleDatasetLatest.Config(
                        data_sources=sources,
                        is_wandb=True,
                        label_type="logits",
                        top_k_logits=20,
                    ),
                    generate_every_n_steps=16,
                    generate_datasets=[
                        mmlu_subset()
                    ],
                    eval_every_n_steps=64,
                    eval_datasets=[
                        # *mtob_eval_datasets(
                        #     local_batch_size=16,
                        # ),
                        # EvalDatasetConfig(
                        #     name_for_wandb="mmlu",
                        #     local_batch_size=16,
                        #     dataset=MMLUEvalDataset.Config(num_samples=128),
                        # ),
                    ],
                    generate_max_new_tokens=64,
                    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                        max_tokens=num_tokens
                    ),
                    loss_type="logits",
                    save_every_n_steps=128,
                    epochs=2,
                    lr=lr,
                    # lr_scheduler=CosWithWarmup.Config(max_steps=5000, warmup_steps=100),
                    wandb=WandBConfig(
                        project="capsules",
                        tags=["cache_tuning", "development"],
                        entity="hazy-research",
                    ),
                    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
                    global_batch_size=bs,
                    local_batch_size=1,
                    # distributed_backend="nccl",
                )
            )

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])
