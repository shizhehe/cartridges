import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.tasks.reglab import ReglabHousingQAGenerateDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"
bs = 64

configs = []

num_tokens = 8192
lr = 5e-3
configs.append(
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_{STATE.lower()}_lr{lr}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        dataset=CapsuleDatasetLatest.Config(
            data_sources=[
                ("hazy-research/capsules/m04d10_generate_reglab_qa_alabama_n32768_nstatutes100:v0", None),
                ("hazy-research/capsules/m04d10_generate_reglab_qa_alabama_n32768_nstatutes100:v0", None),
                
            ],
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,

        ),
        generate_every_n_steps=256,
        generate_max_new_tokens=512,
        generate_datasets=[
            GenerateDatasetConfig(
                dataset=ReglabHousingQAGenerateDataset.Config(
                    states=[STATE], 
                    cot=True,
                    max_questions=None,
                ),
                name_for_wandb=f"housing_qa_generate_{STATE}",
            ),
        ],
        eval_every_n_steps=64,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb=f"housing_qa_new_eval_{STATE}",
                local_batch_size=16,
                dataset=CapsuleDataset.Config(
                    data_sources=[
                        # replace with actual eval dataset once the run is done
                        ("hazy-research/capsules/housing_qa_eval_data_alabama:v7", None),
                    ],
                    is_wandb=True,
                    label_type="tokens",
                ),
                only_eval_rank_0=True,
            ),
        ],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        save_every_n_steps=128,
        epochs=1,
        lr=lr,
        wandb=WandBConfig(
            project="capsules",
            tags=["train", f"reglab_housing_{STATE.lower()}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=4,
    )
)

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)
