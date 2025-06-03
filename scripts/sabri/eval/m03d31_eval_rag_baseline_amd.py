import os
from pathlib import Path

import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.rag_baseline import EvaluateRAGBaselineConfig
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
from cartridges.datasets import CartridgeDataset, CartridgeGenerateDataset

from cartridges.transforms import BM25Transform
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
DOC_NAME = "AMD_2022_10K"
num_tokens = 1024  # [2048, 4096, 2048 * 4]
bs = 64


SYSTEM_PROMPT_TEMPLATE = """Please use the information in the following financial document to answer the user's questions.
<title>
{title}
</title>

<document>
{content}
</document>
"""

configs = []


for topk in [1, 2, 4, 8, 16]:
    for max_tokens_per_chunk in [128, 256, 512, 1024, 2048]:
        rag_transform = BM25Transform.Config(
            k1=1.5,
            b=0.75,
            epsilon=0.25,
            max_tokens_per_chunk=max_tokens_per_chunk,
            top_k=topk,
        )

        _, ppl_evals = get_evals(
            FinanceEvals,
            DOC_NAME,
            num_samples=16,  # RE: it's silly we have to specify this
            version_tag="v1",
            batch_size=1,
            transforms=[rag_transform],
        )
        generate_evals = []


        configs += [
            EvaluateRAGBaselineConfig(
                name=f"{file_name}_k{topk}_t{max_tokens_per_chunk}",
                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),
                generate_datasets=[
                    GenerateDatasetConfig(
                        name_for_wandb="finance-bench-gt",
                        dataset=FinanceBenchGenerateDataset.Config(
                            doc_names=[DOC_NAME],
                        ),
                    ),
                    *generate_evals,
                ],
                eval_datasets=[
                    EvalDatasetConfig(
                        name_for_wandb="finance-ppl-gt",
                        local_batch_size=1,
                        dataset=FinanceBenchEvalDataset.Config(
                            doc_names=[DOC_NAME],
                            cot=False,
                            label_type="tokens",
                            data_sources=[],  # ignore this arg
                        ),
                        only_eval_rank_0=True,
                        dataloader_num_workers=16,
                    ),
                    # EvalDatasetConfig(
                    #     name_for_wandb="mmlu",
                    #     local_batch_size=16,
                    #     dataset=MMLUEvalDataset.Config(num_samples=128),
                    # ),
                    *ppl_evals,
                ],
                generate_max_new_tokens=512,
                wandb=WandBConfig(
                    project="cartridges",
                    tags=["cache_tuning", "development"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
                context=FinanceBenchContextConfig(
                    doc_names=[DOC_NAME], force_single_doc=True
                ),
                system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            )
        ]

if __name__ == "__main__":
    pydrantic.main(configs)
