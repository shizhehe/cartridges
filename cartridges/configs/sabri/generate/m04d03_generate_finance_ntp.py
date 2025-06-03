import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.generate.generators.programmatic_old import NTPConvoGenerator
from capsules.generate.run import GenerateConfig
from capsules.tasks.finance import FinanceBenchContextConfig
from capsules.utils import WandBConfig


file_name = Path(__file__).stem

DOC_NAME = "AMD_2022_10K"

config = GenerateConfig(
    name=FormatStringVariable(
        f"{file_name}_doc{DOC_NAME}_sections{{convo_generator.num_sections_per_convo}}_npreview{{convo_generator.num_preview_tokens}}_max{{convo_generator.max_tokens_per_section}}_{{num_samples}}"
    ),
    convo_generator= NTPConvoGenerator.Config(
        num_sections_per_convo=4,
        num_preview_tokens=64,
        max_tokens_per_section=1024,
    ),
    context=FinanceBenchContextConfig(doc_names=[DOC_NAME], split_on="none"),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32_000,
    batch_size=512,
    max_num_batches_in_parallel=8,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
    parallelism_strategy="process",
)


if __name__ == "__main__":
    pydrantic.main(config)
