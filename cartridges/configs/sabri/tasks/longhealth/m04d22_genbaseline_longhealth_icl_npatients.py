import os
from pathlib import Path

import pydrantic

from capsules.clients.together import TogetherClient
from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    # url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

together_client = TogetherClient.Config(
    model_name="Qwen/QwQ-32B",
)

file_name = Path(__file__).stem

configs = []
SYSTEM_PROMPT_TEMPLATE = f"""Please reference the patient's medical records included below to answer the user's questions.

<patient-records>
{{content}}
</patient-records>
"""

for n_patients in [10]: #, 3, 5, 7, 9, 11]:

    patient_idxs = list(range(1, n_patients + 1))
    patients_str = ''.join(f"p{idx:02d}" for idx in patient_idxs)  # used for names and tags
    patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]
    configs += [
        GenerateBaselineConfig(
            name=f"{file_name}_{patients_str}",
            generator=ICLBaseline.Config(
                client=together_client,
                system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
                temperature=0.0,
                max_completion_tokens=512,
            ),
            dataset=GenerateDatasetConfig(
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=patient_ids, 
                    cot=True,
                ),
                name_for_wandb=f"longhealth_mc",
                num_samples=1,
                temperature=0.0,
            ),
            context=LongHealthContextConfig(
                patient_ids=patient_ids
            ),
            max_num_batches_in_parallel=32,
            batch_size=32,
            wandb=WandBConfig(
                project="capsules",
                tags=[f"longhealth", "genbaseline", f"patients_{patients_str}"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        )
    ]

if __name__ == "__main__":
    pydrantic.main(configs)
