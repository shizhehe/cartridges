import os
from pathlib import Path

import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.tasks.finance import FinanceBenchContextConfig, FinanceBenchEvalDataset
from cartridges.tasks.mmlu import MMLUEvalDataset
from cartridges.train import EvalDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset, CartridgeDatasetWithRefDist

from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
DOC_NAME = "AMD_2022_10K"
bs = 64

configs = []
generate_evals, ppl_evals = get_evals(
    FinanceEvals,
    DOC_NAME,
    num_samples=16,  # RE: it's silly we have to specify this
    version_tag="v1",
    batch_size=16,
)

for num_tokens in [8192]:
    for lr in [5e-2]:
        configs.append(
            TrainConfig(
                name=f"{file_name}_{DOC_NAME}_nt{num_tokens}",
                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),
                dataset=CartridgeDataset.Config(
                    data_sources=[
                        ("hazy-research/Cartridges/m04d08_generate_finance_simple_question_sequential_AMD_2022_10K_n32768:v0", None),
                        ("hazy-research/Cartridges/m04d09_generate_finance_rag_AMD_2022_10K_n32768:v0", None),

                        # sliding window
                        ("hazy-research/Cartridges/m04d08_generate_finance_fair_section_AMD_2022_10K_n32768:v3", None),
                        ("hazy-research/Cartridges/m04d08_generate_finance_fair_section_AMD_2022_10K_n32768:v2", None),
                    ],
                    is_wandb=True,
                    label_type="logits",
                    top_k_logits=20,
                ),
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
                    max_tokens=num_tokens
                ),
                loss_type="logits",
                save_every_n_steps=128,
                epochs=1,
                lr=lr,
                wandb=WandBConfig(
                    project="cartridges",
                    tags=[],
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
