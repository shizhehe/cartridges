import os
from pathlib import Path

import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.tasks.finance import (
    FinanceBenchContextConfig,
    FinanceBenchEvalDataset,
    FinanceBenchGenerateDataset,
)
from capsules.tasks.reglab.housing_qa import (
    HousingStatutesContextConfig,
    HousingEvalAnswerGenerator,
)
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import (
    CapsuleDataset,
    CapsuleDatasetWithRefDist,
    CapsuleGenerateDataset,
)

from capsules.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
# DOC_NAME = "AMD_2022_10K"
STATE = "Alabama"
bs = 64

configs = []

num_tokens = 2048
lr = 1e-2
configs.append(
    TrainConfig(
        name=f"housing_qa_{STATE}_nt{num_tokens}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        dataset=CapsuleDataset.Config(
            data_sources=[
                (
                    # relevant only
                    "hazy-research/capsules/housing_qa_train_data_alabama:v0",
                    None,
                )
            ],
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
            # context=HousingStatutesContextConfig(
            #     states=[STATE],
            # ),
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
        eval_every_n_steps=64,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb=f"housing_qa_train_{STATE}",
                local_batch_size=16,
                dataset=CapsuleDataset.Config(
                    data_sources=[
                        # replace with actual eval dataset once the run is done
                        ("hazy-research/capsules/housing_qa_eval_data_alabama:v0", None),   
                        # ("hazy-research/capsules/housing_qa_alabama:v5", None),
                    ],
                    is_wandb=True,
                    label_type="tokens",
                ),
                only_eval_rank_0=True,
            ),
        ],
        generate_max_new_tokens=512,
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        save_every_n_steps=128,
        epochs=4,
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
    pydrantic.main(configs)
