import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient



from capsules.data.mtob import KalamangSectionedConfig
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generators.programmatic.programmatic_base import ProgrammaticConvoGenerator
from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.run_new import GenerateConfig

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


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(8)],
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)


def config():
    return GenerateConfig(
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=ProgrammaticConvoGenerator.Config(
            convo_type="next_passage",
            system_prompt_template=ANSWER_SYSTEM_PROMPT,
            answer_client=client,
            subcontext_generator=RandomizedSlidingWindowGenerator.Config(
                num_passes=2,
                min_window_size=4,
                max_window_size=6,
                min_step_size=1,
                max_step_size=2,
                seed=32,
            ),
        ),
        context=KalamangSectionedConfig(max_tokens_per_section=5_000,  book_size="medium"),
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        num_samples=32,
        batch_size=1, # ignored
        max_num_batches_in_parallel=32,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
