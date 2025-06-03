import os
from pathlib import Path

import pydrantic

from cartridges.generate_baseline import GenerateBaselineConfig, ICLBaseline
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from cartridges.train import GenerateDatasetConfig

from cartridges.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    # url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem

configs = []
SYSTEM_PROMPT_TEMPLATE = f"""Please reference the patient's medical records included below to answer the user's questions.

<patient-records>
{{content}}
</patient-records>
"""


PATIENT_IDXS = [1, 2, 3]

patients_str = ''.join(f"p{idx:02d}" for idx in PATIENT_IDXS)  # used for names and tags
patient_ids = [f"patient_{idx:02d}" for idx in PATIENT_IDXS]
configs += [
    GenerateBaselineConfig(
        name=f"{file_name}_{patients_str}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.3
        ),
        dataset=GenerateDatasetConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids, 
                cot=True,
            ),
            name_for_wandb=f"longhealth_mc",
            num_samples=16,
            temperature=0.3,
        ),
        context=LongHealthContextConfig(
            patient_ids=patient_ids
        ),
        generate_max_new_tokens=512,
        max_num_batches_in_parallel=32,
        batch_size=16,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"longhealth", "genbaseline", f"patients_{patients_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
