import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig, RAGBaseline
from capsules.retrievers import OpenAIRetriever, BM25Retriever
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.tasks.swde import SWDEContextConfig, SWDEMultipleChoiceGenerateDataset
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem

configs = []
SYSTEM_PROMPT_TEMPLATE = f"""Please reference the website below to answer the questions.

<website>
{{content}}
</website>
"""


configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{html_page}_topk{top_k}_chunk{chunk_size}",
        generator=RAGBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            retriever=BM25Retriever.Config(
                max_tokens_per_chunk=256
            ),
            top_k=top_k,
        ),
        dataset=GenerateDatasetConfig(
            dataset=SWDEMultipleChoiceGenerateDataset.Config(
                webpage_id=html_page, 
                cot=False,
            ),
            name_for_wandb=f"swde_imdb_generate_{html_page}_topk{top_k}_chunk{chunk_size}",
        ),
        context=SWDEContextConfig(
            webpage_id=html_page,
            max_tokens_per_section=-1,
        ),
        generate_max_new_tokens=512,
        max_num_batches_in_parallel=1,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"swde_imdb", "genbaseline", f"webpage_{html_page}_topk{top_k}_chunk{chunk_size}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
    for top_k in [1, 2, 4, 8, 16]
    for html_page in ['0000.htm', '0001.htm', '0002.htm', '0349.htm']
    for chunk_size in [256, 512, 1024, 2048]
]

if __name__ == "__main__":
    pydrantic.main(configs)

