import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.configs.ryan.m03d27.doc_names import DOC_NAMES


from capsules.data.mtob import KalamangSectionedConfig
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generators.rewrite import ReWriteForTranslation
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


file_name = Path(__file__).stem


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(8)],
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)


QUESTION_TEMPLATE = """Please summarize the information in the following passage from the Kalamang to English translation manual.

I'd like you to focus on how this information can be used to translate between Kalamang and English.

<passage>
{passage}
</passage>

Restate the information in the passage and explain how it can be used to translate between Kalamang and English.
Relate it to other information in the Kalamang to English translation manual.
Your answer should be informative and provide information about translating between Kalamang and English.
"""

ANSWER_SYSTEM_PROMPT_TEMPLATE = """Please use the following information from the document to answer the question.
Focus on information about translating between Kalamang and English.

<title>
{title}
</title>

Here is the document:
{context}

First, think step-by-step. Then provide your answer to the question.
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""


def config():
    return GenerateConfig(
        
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=ReWriteForTranslation.Config(
            answer_client=client,
            answer_temperature=0.4,
            answer_max_completion_tokens=400,
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT_TEMPLATE,
            question_template=QUESTION_TEMPLATE,
        ),
        context=KalamangSectionedConfig(
            max_tokens_per_section=100_000, book_size="medium"
        ),
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        num_samples=10_000,
        batch_size=8,
        max_num_batches_in_parallel=625,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
