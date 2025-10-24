import os
from pathlib import Path

import pydrantic

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate_baseline import GenerateBaselineConfig, CartridgeBaseline
from capsules.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.longhealth.context import LongHealthStructuredContextConfig
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-batch-geoff-cartridges-llama-3b-1xa1-e33a9e.modal.run/v1",
    use_modal_endpoint=True,  # NOTE: set to False if using deploy_llama_modal_entry.py
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


file_name = Path(__file__).stem


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{patients_str}",
        generator=CartridgeBaseline.Config(
            client=tokasaurus_client,
            cartridges=[{"id": "wauoq23f", "source": "wandb"}],
            system_prompt_template=None,
            temperature=0.3,
            max_completion_tokens=512,
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
            project="capsules",
            tags=[f"longhealth", "gentoka", f"patients_{patients_str}", "cartridge"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)