import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.generate.context_convo_generators.programmatic import NTPContextConvoGenerator
from cartridges.generate.run import GenerateConfig
from cartridges.tasks.finance import FinanceBenchContextConfig
from cartridges.utils import WandBConfig


file_name = Path(__file__).stem

DOC_NAME = "AMD_2022_10K"

config = GenerateConfig(
    name=FormatStringVariable(
        f"{file_name}_doc{DOC_NAME}_sections{{convo_generator.num_sections_per_convo}}_npreview{{convo_generator.num_preview_tokens}}_max{{convo_generator.max_tokens_per_section}}_{{num_samples}}"
    ),
    convo_generator=NTPContextConvoGenerator.Config(
        num_sections_per_convo=4,
        num_preview_tokens=64,
        max_tokens_per_section=1024,
    ),
    context=FinanceBenchContextConfig(doc_names=[DOC_NAME], force_single_doc=True),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=32_000,
    batch_size=512,
    max_num_batches_in_parallel=8,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
    ),
    parallelism_strategy="process",
)


if __name__ == "__main__":
    pydrantic.main(config)
