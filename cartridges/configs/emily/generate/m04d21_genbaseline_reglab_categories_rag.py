import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline, RAGBaseline
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.retrievers import OpenAIRetriever
from capsules.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.reglab import ReglabHousingCategoriesQAGenerateDataset
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


file_name = Path(__file__).stem

configs = []
STATE = "Alabama"
SYSTEM_PROMPT_TEMPLATE = f"""You are a legal expert in the state of {STATE}. Please use the {STATE} legal statutes to answer the user's questions."""

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
            dataset=ReglabHousingCategoriesQAGenerateDataset.Config(
                states = [STATE],
                cot=True,
            ),
            name_for_wandb=f"reglab_categories_generate_{STATE}",
        ),
        context=HousingStatutesContextConfig(states=[STATE]),
        generate_max_new_tokens=512,
        max_num_batches_in_parallel=16,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"reglab", "genbaseline", "{STATE}", "reglab_categories", "eval"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
    for top_k in [1, 2, 4, 8]
]
if __name__ == "__main__":
    pydrantic.main(configs)
