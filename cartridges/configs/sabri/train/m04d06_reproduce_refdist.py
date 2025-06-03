import os
from pathlib import Path

import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.tasks.finance import FinanceBenchContextConfig, FinanceBenchEvalDataset
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import EvalDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleDatasetWithRefDist

from capsules.utils import WandBConfig

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

for num_tokens in [1024]:
    for lr in [5e-2]:
        configs.append(
            TrainConfig(
                name=f"{file_name}_{DOC_NAME}_nt{num_tokens}",
                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),
                dataset=CapsuleDatasetWithRefDist.Config(
                    data_sources=[
                        ("hazy-research/capsules/m04d04_amd_sliding_window_memorization_refdist:v1", None),
                        ("hazy-research/capsules/m04d04_amd_sliding_window_basic_qa_refdist:v0", None),
                        ("hazy-research/capsules/m04d04_amd_random_window_basic_qa_ref_dist:v0", None),
                        ("hazy-research/capsules/m04d04_amd_random_memorization_refdist:v0", None),
                    ],
                    is_wandb=True,
                    label_type="logits",
                    top_k_logits=20,
                    context=FinanceBenchContextConfig(
                        doc_names=["AMD_2022_10K"], split_on="none"
                    ),
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
                    project="capsules",
                    tags=[],
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
