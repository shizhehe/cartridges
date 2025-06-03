import os
from pathlib import Path

import pydrantic

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.generate_baseline import GenerateBaselineConfig, ICLBaseline, ICLBaselineFirstKTokens
from cartridges.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig
from cartridges.train import GenerateDatasetConfig

from cartridges.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    # url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


file_name = Path(__file__).stem

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the patient medical records included below to answer the user's questions.

<patient-records>
{{content}}
</patient-records>
"""


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{patients_str}",
        generator=ICLBaselineFirstKTokens.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.3,
            max_completion_tokens=512,
            frac_of_tokens=(1 - ratio),
        ),
        dataset=GenerateDatasetConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids, 
                cot=True,
            ),
            name_for_wandb=f"longhealth_mc",
            num_samples=8,
            temperature=0.3,
        ),
        context=LongHealthStructuredContextConfig(patient_ids=patient_ids),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"longhealth", "genbaseline", f"patients_{patients_str}", "icl"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    )
    for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
]

if __name__ == "__main__":
    pydrantic.main(configs)
