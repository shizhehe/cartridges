import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.reglab import ReglabHousingCategoriesQAGenerateDataset
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.tasks.reglab.utils import ALL_STATES
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

# tokasaurus_client = TokasaurusClient.Config(
#     url="https://hazyresearch--tksrs-batch-main-llama-8b-1xh100-min0-serve.modal.run",
#     use_modal_endpoint=True,
#     model_name="meta-llama/Llama-3.2-8B-Instruct",
# )

file_name = Path(__file__).stem

configs = []
NUM_STATES = 20
states = ALL_STATES[:NUM_STATES]

SYSTEM_PROMPT_TEMPLATE = f"""You are expert on housing law in the United States. Please use the legal statutes below to answer the user's questions."""

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{NUM_STATES}states",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.0,
            max_completion_tokens=1024,
        ),
        dataset=GenerateDatasetConfig(
            dataset=ReglabHousingCategoriesQAGenerateDataset.Config(
                states = states,
                cot=True,
            ),
            name_for_wandb=f"reglab_categories_generate",
            temperature=0.0,
            num_samples=1
        ),
        context=HousingStatutesContextConfig(
            states=states,
            split_on="statute",
            num_extra_statutes=0
        ),
        max_num_batches_in_parallel=4,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"reglab-housing", "genbaseline", f"{NUM_STATES}states", "categories"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
