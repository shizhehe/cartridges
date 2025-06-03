import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.clients.openai import OpenAIClient
from capsules.tasks.novelqa import NovelContextConfig, NovelMultipleChoiceGenerateDataset
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

openai_client = OpenAIClient.Config()

file_name = Path(__file__).stem

configs = []
BOOK_IDXS = 'Frankenstein_Demo'
SYSTEM_PROMPT_TEMPLATE = f"""Please reference the novel below to answer the questions.

<novel>
{{content}}
</novel>
"""

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{BOOK_IDXS}",
        generator=ICLBaseline.Config(
            client=openai_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        ),
        dataset=GenerateDatasetConfig(
            dataset=NovelMultipleChoiceGenerateDataset.Config(
                book_id=BOOK_IDXS, 
                cot=False,
            ),
            name_for_wandb=f"novelqa_generate_{BOOK_IDXS}",
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
            tags=[f"novelqa", "genbaseline", f"books_{BOOK_IDXS}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)

