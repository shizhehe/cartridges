import os
from pathlib import Path

import pydrantic

from cartridges.clients.openai import  OpenAIClient
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.evaluate import CartridgeBaseline, EvaluateConfig, CartridgeConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.evaluate import GenerationEvalConfig
from cartridges.utils import WandBConfig


# client = OpenAIClient.Config(
#     base_url="http://localhost:10210/v1/cartridge",
#     model_name="Qwen/Qwen3-8b",
# )
client = TokasaurusClient.Config(
    url="http://localhost:10210/",
    model_name="Qwen/Qwen3-8b",
)


file_name = Path(__file__).stem


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

configs = [
    EvaluateConfig(
        name=f"{file_name}_{patients_str}",
        generator=CartridgeBaseline.Config(
            client=client,
             cartridges=[CartridgeConfig(
              id="1wqgicqv",
              source="wandb",
              force_redownload=False
            )],
            temperature=0.3,
            max_completion_tokens=2048,
            context=LongHealthResource.Config(
                patient_ids=patient_ids,
            ),
        ),
        eval=GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids, 
                cot=True,
            ),
            name_for_wandb=f"longhealth_mc",
            num_samples=1,
            temperature=0.3,
        ),
        max_num_batches_in_parallel=4,
        batch_size=32,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"longhealth", "genbaseline", f"patients_{patients_str}", "icl"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
