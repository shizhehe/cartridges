import os
from pathlib import Path

import pydrantic

from cartridges.clients.openai import  OpenAIClient
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
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
    model_name="Qwen/Qwen3-4b",
    base_timeout=160,
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
            # cartridges=[
            #     # CartridgeConfig(
            #     #     id="1w9e5agx",
            #     #     source="wandb",
            #     #     force_redownload=False
            #     # ),
            #     # CartridgeConfig(
            #     #     id="18xjajjv",
            #     #     source="wandb",
            #     #     force_redownload=False
            #     # ),

            #     # Longhealth 2048: hazy-research/cartridges/e01tf9rv
            #     # CartridgeConfig(
            #     #     id="e01tf9rv",
            #     #     source="wandb",
            #     #     force_redownload=False
            #     # ),
            #     # MTOB 2048: hazy-research/cartridges/qzwkmr28
            #     # CartridgeConfig(
            #     #     id="qzwkmr28",
            #     #     source="wandb",
            #     #     force_redownload=False
            #     # ),
            # ],
            enable_thinking=False,
            temperature=0.0,
            max_completion_tokens=2048,
            context=LongHealthResource.Config(
                patient_ids=patient_ids,
            ),
        ),
        eval=GenerationEvalConfig(
            name_for_wandb=f"mmtob-ke-test",
            dataset=MTOBKalamangToEnglishGenerateDataset.Config(use_cot=False),
            batch_size=16,
            generate_max_new_tokens=128,
            num_samples=1,
            temperature=0.0,
        ),
        max_num_batches_in_parallel=1,
        batch_size=64,
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
