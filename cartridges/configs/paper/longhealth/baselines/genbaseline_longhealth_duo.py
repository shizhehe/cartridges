import pydrantic
from dataclasses import dataclass
import os
from typing import Literal

from capsules.generate_baseline import (
    GenerateBaselineConfig,
    KVCacheCompressionBaseline,
)
from capsules.tasks.longhealth import LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.longhealth.context import LongHealthStructuredContextConfig
from capsules.tasks.mtob import (
    MtobKalamangToEnglishGenerateDataset,
    MtobEnglishToKalamangGenerateDataset,
)
from capsules.tasks.mtob.context import MTOBNoStructuredContext
from capsules.train import GenerateDatasetConfig
from capsules.utils.wandb import WandBConfig


@dataclass
class KVCompression:
    compression_method: Literal["duo_on_the_fly", "expected_attention"]
    compression_ratio: float


compression_settings = [
    KVCompression(
        compression_method="duo_on_the_fly",
        compression_ratio=ratio,
    )
    for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
]
# ] + [
#     KVCompression(
#         compression_method="expected_attention",
#         compression_ratio=ratio,
#     )
#     for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
# ]

NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]


configs = [
    GenerateBaselineConfig(
        name=f"longheath_3b_{compression.compression_method}_ratio{compression.compression_ratio}",
        generator=KVCacheCompressionBaseline.Config(
            kv_compression=compression.compression_method,
            kv_compression_ratio=compression.compression_ratio,
            model="Meta-Llama/Llama-3.2-3B-Instruct",
            max_completion_tokens=512,
        ),
        dataset=GenerateDatasetConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids,
                cot=True,
            ),
            name_for_wandb=f"longhealth_mc",
            num_samples=1,
            num_samples_final=1,
            batch_size=1,
            temperature=0.0,
        ),
        context=LongHealthStructuredContextConfig(patient_ids=patient_ids),
        max_num_batches_in_parallel=1,
        batch_size=1,
        wandb=WandBConfig(
            project="capsules",
            tags=["longhealth", "kv-press", f"patients_{patients_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
    for compression in compression_settings
]


if __name__ == "__main__":
    pydrantic.main(configs)
