from dataclasses import dataclass
import os
from typing import Literal

from capsules.generate_baseline import (
    GenerateBaselineConfig,
    KVCacheCompressionBaseline,
)
from capsules.tasks.mtob import (
    MtobKalamangToEnglishGenerateDataset,
    MtobEnglishToKalamangGenerateDataset,
)
from capsules.tasks.mtob.context import MTOBNoStructuredContext
from capsules.train import GenerateDatasetConfig
from capsules.utils.wandb import WandBConfig


@dataclass
class KVCompression:
    compression_method: Literal["duo", "expected_attention", "duo_on_the_fly", "snapkv", "tova", 'key_diff']
    compression_ratio: float


all_compression_settings = {
    "duo": [
        KVCompression(
            compression_method="duo_on_the_fly",
            compression_ratio=ratio,
        )
        for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ],
    "expected_attention": [
        KVCompression(
            compression_method="expected_attention",
            compression_ratio=ratio,
        )
        for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ],

    "tova": [
        KVCompression(
            compression_method="tova",
            compression_ratio=ratio,
        )
        for ratio in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    ],

    "key_diff": [
        KVCompression(
            compression_method="key_diff",
            compression_ratio=ratio,
        )
        for ratio in [0.0, 0.5]
    ],

    "tova": [
        KVCompression(
            compression_method="tova",
            compression_ratio=ratio,
        )
        for ratio in [0.01]
    ],

}



def get_configs(
    name,
    model,
    direction: Literal["ke", "ek"],
    attention_type: Literal["duo", "expected_attention", "key_diff", "tova"],
):
    compression_settings = all_compression_settings[attention_type]

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
                project="capsules",
                tags=[],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        )
        for compression in compression_settings
    ]
