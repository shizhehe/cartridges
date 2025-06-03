import os
from pathlib import Path

import pydrantic

from cartridges.generate_baseline import GenerateBaselineConfig
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.tasks.reglab import ReglabHousingQAGenerateDataset
from cartridges.tasks.reglab.housing_qa import HousingStatutesContextConfig
from cartridges.train import GenerateDatasetConfig
from cartridges.tasks.reglab.baseline import ReglabHousingGoldPassageBaseline

from cartridges.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="http://localhost:8889/v1",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"
NUM_STATUTES = None
SYSTEM_PROMPT_TEMPLATE = f"""Please use the {STATE} state code included below to answer the user's questions.

<state_code>
{{content}}
</state_code>
"""

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{STATE}",
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
            states=[STATE],
            split_on="statute",
            num_statutes=NUM_STATUTES,
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
