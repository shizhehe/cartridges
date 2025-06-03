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
)
from capsules.tasks.mmlu import MMLUEvalDataset
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
    for num_tokens in [6144]:
        for lr in [0.01]:
            sources = [
                (
                    "/scr/ryanehrlich/capsules/data_dir/2025-05-03-18-06-35-m05d03_generate_mtob_auto/54c50871-07c9-4440-bf32-09a2d039052d/artifact/dataset.pkl",
                    None,
                ),
                (
                    "/scr/ryanehrlich/capsules/data_dir/2025-05-03-19-25-40-m05d03_generate_mtob_auto/068b8c42-f03e-4e90-b01e-edfd786dfb71/artifact/dataset.pkl",
                    None
                )
            ]
            configs.append(
                TrainConfig(
                    name=f"{file_name}_nt{num_tokens}",
                    model=HFModelConfig(
                        pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
                    ),
                    dataset=CapsuleDatasetLatest.Config(
                        data_sources=sources,
                        is_wandb=True,
                        label_type="logits",
                        top_k_logits=20,
                    ),
                    generate_every_n_steps=64,
                    generate_datasets=[
                        *mtob_generate_datasets(),
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
                    epochs=3,
                    lr=lr,
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
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])
