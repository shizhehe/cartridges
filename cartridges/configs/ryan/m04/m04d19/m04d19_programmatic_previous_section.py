import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient



from capsules.data.mtob import KalamangSectionedConfig
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generators.programmatic.programmatic_base import ProgrammaticConvoGenerator
from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.generate_training import GenerateConfig

from pydrantic.variables import FormatStringVariable


from capsules.generate.subcontext import (
    RandomSubcontextGenerator,
    RandomizedSlidingWindowGenerator,
)

from capsules.tasks.finance import (
    FinanceBenchContextConfig,
    FinanceBenchSectionedContextConfig,
)
from capsules.utils import WandBConfig

ANSWER_SYSTEM_PROMPT = """Please use the following information from the document to answer the question.
<title>
{title}
</title>

Here is the document:
{subcontext}
"""

file_name = Path(__file__).stem



def config():
    return GenerateConfig(
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=ProgrammaticConvoGenerator.Config(
            convo_type="previous_passage",
            tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        ),
        context=KalamangSectionedConfig(max_tokens_per_section=10_000,  book_size="long"),
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        num_samples=1,
        batch_size=1, # ignored
        max_num_batches_in_parallel=1,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
