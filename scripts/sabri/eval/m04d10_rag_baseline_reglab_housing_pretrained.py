import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.initialization.strategies.pretrained import KVCacheFromPretrained
from cartridges.tasks.reglab import ReglabHousingQAGenerateDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"
bs = 64

configs = []

num_tokens = 2048
lr = 5e-3
configs.append(
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_{STATE.lower()}_lr{lr}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        dataset=CartridgeDataset.Config(
            data_sources=[
                # ("hazy-research/Cartridges/m04d09_generate_reglab_rag_alabama_n32768:v4", None),
                ("hazy-research/Cartridges/m04d09_generate_reglab_rag_alabama_n32768:v5", None),

                ("hazy-research/Cartridges/m04d10_generate_reglab_fair_section_alabama_n32768:v0", None),
            ],
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,

        ),
        eval_every_n_steps=64,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb=f"housing_qa_new_eval_{STATE}",
                local_batch_size=16,
                dataset=CartridgeDataset.Config(
                    data_sources=[
                        # replace with actual eval dataset once the run is done
                        ("hazy-research/Cartridges/housing_qa_eval_data_alabama:v7", None),
                    ],
                    is_wandb=True,
                    label_type="tokens",
                ),
                only_eval_rank_0=True,
            ),
        ],
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
        kv_cache_initializer=KVCacheFromPretrained.Config(
            wandb_run_id="hazy-research/Cartridges/i304hlbh",
        ),
        loss_type="logits",
        save_every_n_steps=128,
        epochs=0,
        lr=lr,
        wandb=WandBConfig(
            project="cartridges",
            tags=["train", f"reglab_housing_{STATE.lower()}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=4,
    )
)

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)
