import os
from pathlib import Path

import pydrantic

from cartridges.generate_baseline import GenerateBaselineConfig, ICLBaseline
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from cartridges.tasks.reglab import ReglabHousingCategoriesQAGenerateDataset
from cartridges.tasks.reglab.housing_qa import HousingStatutesContextConfig
from cartridges.train import GenerateDatasetConfig

from cartridges.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem

configs = []
STATE = "Alabama"
SYSTEM_PROMPT_TEMPLATE = f"""You are a legal expert in the state of {STATE}. Please use the {STATE} legal statutes to answer the user's questions."""

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{STATE}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.3
        ),
        dataset=GenerateDatasetConfig(
            dataset=ReglabHousingCategoriesQAGenerateDataset.Config(
                states = states,
                cot=True,
            ),
            name_for_wandb=f"reglab_categories_generate_{STATE}",
            temperature=0.3,
            num_samples=16
        ),
        context=HousingStatutesContextConfig(
            states=[STATE]
        ),
        generate_max_new_tokens=1024,
        max_num_batches_in_parallel=1,
        batch_size=16,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"reglab-housing", "genbaseline", f"{STATE}", "reglab-categories"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
