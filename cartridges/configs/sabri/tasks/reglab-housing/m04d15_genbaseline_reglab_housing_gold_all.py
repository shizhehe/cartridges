import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.tasks.reglab import ReglabHousingQAGenerateDataset
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.train import GenerateDatasetConfig
from capsules.tasks.reglab.baseline import ReglabHousingGoldPassageBaseline

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="http://localhost:8889/v1",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

file_name = Path(__file__).stem

configs = []
STATE = None
NUM_STATUTES = None
SYSTEM_PROMPT_TEMPLATE = f"""Please use the {STATE} state code included below to answer the user's questions.

<state_code>
{{content}}
</state_code>
"""

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_all",
        generator=ReglabHousingGoldPassageBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        ),
        dataset=GenerateDatasetConfig(
            dataset=ReglabHousingQAGenerateDataset.Config(
                states=[STATE], 
                cot=True,
            ),
            name_for_wandb=f"housing_qa_generate_{STATE}",
        ),
        context=HousingStatutesContextConfig(
            states=[],
            split_on="statute",
            num_statutes=NUM_STATUTES,
        ),
        generate_max_new_tokens=512,
        max_num_batches_in_parallel=16,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"reglab_housing_all", "eval"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
