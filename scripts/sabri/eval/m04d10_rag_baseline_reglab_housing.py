import os
from pathlib import Path

import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.rag_baseline import EvaluateRAGBaselineConfig
from cartridges.retrievers import BM25Retriever, OpenAIRetriever
from cartridges.transforms import RetrieverTransform
from cartridges.tasks.reglab.housing_qa import HousingStatutesContextConfig
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset
from cartridges.context import BaseContextConfig
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

STATE = "Alabama"

context = HousingStatutesContextConfig(
    states=[STATE],
    split_on="statute"
)

RETRIEVER = os.environ.get("RETRIEVER", "bm25")


for topk in [1, 2, 4, 8, 16]:
    for max_tokens_per_chunk in [128, 256]:

        if RETRIEVER == "bm25":
            retriever = BM25Retriever.Config(
                k1=1.5,
                b=0.75,
                epsilon=0.25,
                max_tokens_per_chunk=max_tokens_per_chunk,
            )
        elif RETRIEVER == "openai":
            retriever = OpenAIRetriever.Config(
                embedding_model="text-embedding-3-large",
                max_tokens_per_chunk=max_tokens_per_chunk,
            )

        rag_transform = RetrieverTransform.Config(
            retriever=retriever,
            context=context,
            top_k=topk,
        )

        configs += [
            EvaluateRAGBaselineConfig(
                name=f"{file_name}_{RETRIEVER}_k{topk}_t{max_tokens_per_chunk}",
                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),
                eval_datasets=[
                    EvalDatasetConfig(
                        name_for_wandb=f"housing_qa_new_eval_{STATE}",
                        local_batch_size=1,
                        dataset=CartridgeDataset.Config(
                            data_sources=[
                                # replace with actual eval dataset once the run is done
                                ("hazy-research/Cartridges/housing_qa_eval_data_alabama:v7", None),
                            ],
                            is_wandb=True,
                            label_type="tokens",
                            convo_transforms=[rag_transform],
                        ),
                        only_eval_rank_0=True,
                    ),
                ],
                generate_max_new_tokens=512,
                wandb=WandBConfig(
                    project="cartridges",
                    tags=[f"reglab_housing_{STATE.lower()}", "eval"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
                context=context,
                system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            )
        ]

if __name__ == "__main__":
    pydrantic.main(configs)
