from dataclasses import dataclass
import os
from typing import Literal

from cartridges.generate_baseline import (
    GenerateBaselineConfig,
    KVCacheCompressionBaseline,
)
from cartridges.tasks.mtob import (
    MtobKalamangToEnglishGenerateDataset,
    MtobEnglishToKalamangGenerateDataset,
)
from cartridges.tasks.mtob.context import MTOBNoStructuredContext
from cartridges.train import GenerateDatasetConfig
from cartridges.utils.wandb import WandBConfig


@dataclass
class KVCompression:
    compression_method: Literal["duo", "expected_attention", "duo_on_the_fly"]
    compression_ratio: float


compression_settings_duo = [
    KVCompression(
        compression_method="duo_on_the_fly",
        compression_ratio=ratio,
    )
    for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
]
compression_settings_expected_attention = [
    KVCompression(
        compression_method="expected_attention",
        compression_ratio=ratio,
    )
    for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
]


def get_configs(
    name,
    model,
    direction: Literal["ke", "ek"],
    attention_type: Literal["duo", "expected_attention"],
):
    compression_settings = (
        compression_settings_duo
        if attention_type == "duo_on_the_fly"
        else compression_settings_expected_attention
    )

    if direction == "ke":
        dataset = GenerateDatasetConfig(
            name_for_wandb=f"mmtob-ke-test",
            dataset=MtobKalamangToEnglishGenerateDataset.Config(use_cot=False),
            batch_size=1,
        )
    elif direction == "ek":
        dataset = GenerateDatasetConfig(
            name_for_wandb=f"mmtob-ek-test",
            dataset=MtobEnglishToKalamangGenerateDataset.Config(use_cot=False),
            batch_size=1,
        )
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'ke' or 'ek'.")

    return [
        GenerateBaselineConfig(
            name=f"{name}_{compression.compression_method}_ratio{compression.compression_ratio}",
            generator=KVCacheCompressionBaseline.Config(
                kv_compression=compression.compression_method,
                kv_compression_ratio=compression.compression_ratio,
                model=model,
            ),
            dataset=dataset,
            context=MTOBNoStructuredContext(setup="medium_and_sentences"),
            max_num_batches_in_parallel=1,
            batch_size=16,
            wandb=WandBConfig(
                project="cartridges",
                tags=[],
                entity="hazy-research",
            ),
            output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        )
        for compression in compression_settings
    ]
