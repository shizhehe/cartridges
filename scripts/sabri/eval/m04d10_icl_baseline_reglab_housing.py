import os
from pathlib import Path

import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.icl_baseline import ICLBaselineConfig
from cartridges.tasks.reglab import ReglabHousingQAGenerateDataset
from cartridges.tasks.reglab.housing_qa import HousingStatutesContextConfig
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset

from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"

SYSTEM_PROMPT_TEMPLATE = """Please use the information in the following legal document to answer the user's questions.
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
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        generate_every_n_steps=128,
        generate_datasets=[
            GenerateDatasetConfig(
                dataset=ReglabHousingQAGenerateDataset.Config(
                    states=[STATE], 
                    cot=True,
                    max_questions=None,
                ),
                name_for_wandb=f"housing_qa_generate_{STATE}",
            ),
        ],
        eval_every_n_steps=32,
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
                ),
                only_eval_rank_0=True,
            ),
        ],
        context=HousingStatutesContextConfig(
            states=[STATE],
            split_on="statute"
        ),
        generate_max_new_tokens=512,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"reglab_housing_{STATE.lower()}", "eval"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],

        
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
