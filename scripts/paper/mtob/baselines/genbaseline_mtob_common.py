from dataclasses import dataclass
import os
from typing import List, Literal

from cartridges.generate_baseline import (
    GenerateBaselineConfig,
    BaselineGenerator,
    KVCacheCompressionBaseline,
)
from cartridges.tasks.mtob import (
    MtobKalamangToEnglishGenerateDataset,
    MtobEnglishToKalamangGenerateDataset,
)
from cartridges.tasks.mtob.context import MTOBNoStructuredContext
from cartridges.train import GenerateDatasetConfig
from cartridges.utils.wandb import WandBConfig


def get_configs(
    names: List[str],
    generators: List[BaselineGenerator.Config],
    context: Literal["latex_and_sentences", "medium_and_sentences"] = "medium_and_sentences",
    direction: Literal["ke", "ek"] = "ke",
    max_num_batches_in_parallel: int = 1,
    batch_size: int = 16,
):

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
            name=name,
            generator=generator,
            dataset=dataset,
            context=MTOBNoStructuredContext(setup=context),
            max_num_batches_in_parallel=max_num_batches_in_parallel,
            batch_size=batch_size,
            wandb=WandBConfig(
                project="cartridges",
                tags=["mtob", "genbaseline"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        )
        for name, generator in zip(names, generators)
    ]
