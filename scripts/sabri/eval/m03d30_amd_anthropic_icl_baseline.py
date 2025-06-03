import os
from pathlib import Path

import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.icl_baseline import ICLBaselineConfig
from cartridges.initialization.strategies.first_n_tokens import (
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

from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
DOC_NAME = "AMD_2022_10K"
num_tokens = 1024  # [2048, 4096, 2048 * 4]
bs = 64

configs = []
if True:
    generate_evals, ppl_evals = get_evals(
        FinanceEvals,
        DOC_NAME,
        num_samples=16,  # RE: it's silly we have to specify this
        version_tag="v1",
        batch_size=1,
    )
    generate_evals = []

    configs.append(
        ICLBaselineConfig(
            name=f"{file_name}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),
            generate_every_n_steps=128,
            generate_datasets=[
                # GenerateDatasetConfig(
                #     name_for_wandb="finance-bench-gt",
                #     dataset=FinanceBenchGenerateDataset.Config(
                #         doc_names=[DOC_NAME],
                #     ),
                # ),
                # *generate_evals,
            ],
            eval_every_n_steps=32,
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
            system_prompt_template="""Please use the information in the following financial document to answer the user's questions.
<title>
{title}
</title>

<document>
{content}
</document>
""",
        )
    )

if __name__ == "__main__":
    pydrantic.main(configs)
