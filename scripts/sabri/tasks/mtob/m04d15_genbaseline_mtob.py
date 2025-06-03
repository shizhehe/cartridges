import os
from pathlib import Path

import pydrantic

from cartridges.generate_baseline import GenerateBaselineConfig, ICLBaseline
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.tasks.mtob import MtobKalamangToEnglishGenerateDataset
from cartridges.data.mtob import KalamangContextConfig
from cartridges.train import GenerateDatasetConfig

from cartridges.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="http://localhost:8889/v1",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"
NUM_STATUTES = None
SYSTEM_PROMPT_TEMPLATE = """Please use the detils included below to perform the task of translating the sentence from Kalamang to English.
{content}
"""


configs = [
    GenerateBaselineConfig(
        name=f"{file_name}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            # user_prompt_template="Please translate the following sentence from Kalamang to English: {content}. You can think step-by-step, but eventually output the English sentence between <answer> and </answer> tags.",
            user_prompt_template="Please translate the following sentence from Kalamang to English: {content}. Output the English sentence between <answer> and </answer> tags, do not include any other text.",
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
            project="cartridges",
            tags=[f"reglab_housing_{STATE.lower()}", "eval"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
