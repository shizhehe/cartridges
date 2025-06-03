import os
from pathlib import Path

import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.data.mtob import KalamangContextConfig, KalamangSectionedConfig
from capsules.icl_baseline import ICLBaselineConfig
from capsules.tasks.mtob import mtob_eval_datasets, mtob_generate_datasets
from capsules.tasks.reglab import ReglabHousingQAGenerateDataset
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset

from capsules.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"

SYSTEM_PROMPT_TEMPLATE = """Please use the information in the document to answer the user's questions.
<title>
{title}
</title>

<document>
{content}
</document>
"""

configs = [
    ICLBaselineConfig(
        name=f"{file_name}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
        ),
        generate_every_n_steps=128,
        generate_datasets=[
            *mtob_generate_datasets()
            ][:0],
        eval_every_n_steps=32,
        eval_datasets=[
            *mtob_eval_datasets(
                local_batch_size=1,
            ),
        ],
        context=KalamangSectionedConfig(
            max_tokens_per_section=100_000, book_size="medium"
        ),
        generate_max_new_tokens=512,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"reglab_housing_{STATE.lower()}", "eval"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
