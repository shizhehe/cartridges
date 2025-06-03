import os
from pathlib import Path

import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.kv_initialization.strategies.pretrained import KVCacheFromPretrained
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

configs = []
bs = 32

configs = []
if True:
    for num_tokens, lr in [(4096, 0.02)]:
        configs.append(
            TrainConfig(
                name=f"{file_name}_nt{num_tokens}",
                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
                ),
                dataset=CapsuleDatasetLatest.Config(
                    data_sources=[
                        ("hazy-research/capsules/m04d19_programmatic_next_section_n1:v2", None),
                        ("hazy-research/capsules/m04d19_programmatic_previous_section_n1:v1", None),
                        ("hazy-research/capsules/m04d19_programmatic_unshuffle_n1:v1", None),
                        ("hazy-research/capsules/m04d19_programmatic_unmask_n1:v1", None),
                    ],
                    is_wandb=True,
                    label_type="logits",
                    top_k_logits=1,
                ),
                generate_every_n_steps=32,
                generate_datasets=[
                    *mtob_generate_datasets(),
                ],
                eval_every_n_steps=32,
                eval_datasets=[
                    *mtob_eval_datasets(
                        local_batch_size=16,
                    ),
                    # EvalDatasetConfig(
                    #     name_for_wandb="mmlu",
                    #     local_batch_size=16,
                    #     dataset=MMLUEvalDataset.Config(num_samples=128),
                    # ),
                ],
                generate_max_new_tokens=64,
                # kv_cache_initializer=KVCacheFromPretrained.Config(
                #     wandb_run_id="hazy-research/capsules/cpgaah99"
                # ),
                kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                    max_tokens=num_tokens
                ),
                loss_type="logits",
                save_every_n_steps=128,
                epochs=20,
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
    # config_idx = int(os.environ.get("CFG_IDX", default=0))
    config_idx = 0
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])
