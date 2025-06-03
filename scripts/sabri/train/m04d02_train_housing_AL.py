import os
from pathlib import Path

import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.tasks.finance import (
    FinanceBenchContextConfig,
    FinanceBenchEvalDataset,
    FinanceBenchGenerateDataset,
)
from cartridges.tasks.reglab.housing_qa import (
    HousingStatutesContextConfig,
    HousingEvalAnswerGenerator,
)
from cartridges.tasks.mmlu import MMLUEvalDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import (
    CartridgeDataset,
    CartridgeDatasetWithRefDist,
    CartridgeGenerateDataset,
)

from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
# DOC_NAME = "AMD_2022_10K"
STATE = "Alabama"
bs = 64

configs = []

num_tokens = 2048
lr = 1e-1
configs.append(
    TrainConfig(
        name=f"housing_qa_{STATE}_nt{num_tokens}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        dataset=CartridgeDataset.Config(
            data_sources=[
                (
                    "hazy-research/Cartridges/housing_qa_alabama:v5",
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
                name_for_wandb=f"housing_qa_eval_{STATE}",
                local_batch_size=16,
                dataset=CartridgeDataset.Config(
                    data_sources=[
                        # replace with actual eval dataset once the run is done
                        ("hazy-research/Cartridges/housing_qa_eval_data_alabama:v0", None),   
                    ],
                    is_wandb=True,
                    label_type="tokens",
                ),
            ),
        ],
        generate_max_new_tokens=512,
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        save_every_n_steps=128,
        epochs=1,
        lr=lr,
        wandb=WandBConfig(
            project="cartridges",
            tags=["cache_tuning", "development"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=1,
    )
)

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)
