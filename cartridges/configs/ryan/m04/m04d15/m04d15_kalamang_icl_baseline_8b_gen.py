import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.tasks.mtob import MtobKalamangToEnglishGenerateDataset
from capsules.data.mtob import KalamangContextConfig
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="http://localhost:8889/v1",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"
SYSTEM_PROMPT_TEMPLATE = """Please use the detils to help the user.
{content}
"""


configs = [
    GenerateBaselineConfig(
        name=f"{file_name}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            # user_prompt_template="Please translate the following sentence from Kalamang to English: {content}. You can think step-by-step, but eventually output the English sentence between <answer> and </answer> tags.",
            user_prompt_template="Please translate the following sentence from Kalamang to English: {content}. Output the English sentence and do not include any other text.",
        ),
        dataset=GenerateDatasetConfig(
            name_for_wandb="mmtob-kalamang-to-english",
            dataset=MtobKalamangToEnglishGenerateDataset.Config(

            ),
        ),
        context=KalamangContextConfig(
            book_type="medium",
        ),
        generate_max_new_tokens=512,
        max_num_batches_in_parallel=16,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=["eval"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)