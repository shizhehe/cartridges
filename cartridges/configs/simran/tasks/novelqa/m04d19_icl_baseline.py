import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.tasks.novelqa import NovelContextConfig, NovelMultipleChoiceGenerateDataset
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem

configs = []
BOOK_IDXS = 'Frankenstein_Demo'
SYSTEM_PROMPT_TEMPLATE = f"""Please reference the novel below to answer the questions.

<novel>
{{content}}
</novel>
"""

books_str = BOOK_IDXS
configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{books_str}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        ),
        dataset=GenerateDatasetConfig(
            dataset=NovelMultipleChoiceGenerateDataset.Config(
                book_id=BOOK_IDXS, 
                cot=False,
            ),
            name_for_wandb=f"novelqa_generate_{books_str}",
        ),
        context=NovelContextConfig(
            book_id=BOOK_IDXS,
            max_tokens_per_section=-1,
        ),
        generate_max_new_tokens=512,
        max_num_batches_in_parallel=1,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"novelqa", "genbaseline", f"books_{books_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)

