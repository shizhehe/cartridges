import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.tasks.reglab import ReglabHousingQAGenerateDataset
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.train import GenerateDatasetConfig
from capsules.retrievers import OpenAIRetriever
from capsules.generate_baseline import RAGBaseline

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="http://localhost:8889/v1",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"
NUM_STATUTES = None
SYSTEM_PROMPT_TEMPLATE = f"""Please use the portions of the {STATE} state code included below to answer the user's questions.

<state_code>
{{content}}
</state_code>
"""

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{STATE}_topk{top_k}",
        generator=RAGBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            retriever=OpenAIRetriever.Config(
                embedding_model="text-embedding-3-large",
                max_tokens_per_chunk=256
            ),
            top_k=top_k,
        ),
        dataset=GenerateDatasetConfig(
            dataset=ReglabHousingQAGenerateDataset.Config(
                states=[STATE], 
                cot=True,
            ),
            name_for_wandb=f"housing_qa_generate_{STATE}",
        ),
        context=HousingStatutesContextConfig(
            states=[STATE],
            split_on="statute",
            num_statutes=NUM_STATUTES,
        ),
        generate_max_new_tokens=512,
        max_num_batches_in_parallel=16,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"reglab_housing_{STATE.lower()}", "eval"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
    for top_k in [1, 2, 4, 8]
]

if __name__ == "__main__":
    pydrantic.main(configs)
