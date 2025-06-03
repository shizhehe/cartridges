import os
from pathlib import Path

import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.tasks.finance import (
    FinanceBenchContextConfig,
    FinanceBenchEvalDataset,
    FinanceBenchGenerateDataset,
)
from cartridges.tasks.mmlu import MMLUEvalDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import (
    CartridgeDataset,
    CartridgeDatasetWithRefDist,
    CartridgeGenerateDataset,
)

from cartridges.transforms import BM25Transform
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
DOC_NAME = "AMD_2022_10K"
bs = 64

configs = []
if True:
    rag_transform = BM25Transform.Config(
        k1=1.5,
        b=0.75,
        epsilon=0.25,
        max_tokens_per_chunk=256,
        top_k=4,
    )
    generate_evals, ppl_evals = get_evals(
        FinanceEvals,
        DOC_NAME,
        num_samples=16,  # RE: it's silly we have to specify this
        version_tag="v1",
        batch_size=16,
        transforms=[rag_transform],
    )

    for num_tokens in [8192]:
        for lr in [5e-3]:
            configs.append(
                TrainConfig(
                    name=f"{file_name}_{DOC_NAME}_nt{num_tokens}_rag",
                    model=HFModelConfig(
                        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                    ),
                    dataset=CartridgeDatasetWithRefDist.Config(
                        data_sources=[
                            (
                                "hazy-research/Cartridges/m03d27_gen_simple_data_amd_2022_10k_ref_dist:v2",
                                None,
                            )
                        ],
                        is_wandb=True,
                        label_type="logits",
                        top_k_logits=20,
                        context=FinanceBenchContextConfig(
                            doc_names=["AMD_2022_10K"], force_single_doc=True
                        ),
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
