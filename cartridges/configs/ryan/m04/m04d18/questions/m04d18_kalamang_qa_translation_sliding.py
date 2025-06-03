import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient



from capsules.data.mtob import KalamangSectionedConfig
from capsules.generate.chunk import SimpleCharacterChunker
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

Here is a subset of the document:
{subcontext}


First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""

QUESTION_SYSTEM_PROMPT_TEMPLATE = """The following is excerpts from a collection of documents that should help you learn the language Kalamang.
I'd like you to generate a question from the following context related to understanding the Kalamang language.
This question should be focused on understanding Kalamang for translating it to and from English.

Here is the relevant context:
{subcontext}
"""

QUESTION_PROMPT_TEMPLATE = """Please generate a question that tests the understanding of the Kalamang language for the purposes of translating between English and Kalamang.

Please center your question around this passage:
<passage>
{chunk}
</passage>

Your question does not need to exclusively focus on this passage, but should be related to it.
Further, your question should be open-ended and require many paragraphs to answer.

The question should test the knowledge of the information when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question.
"""
file_name = Path(__file__).stem


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(8)],
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


def config():
    return GenerateConfig(
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=SimpleQuestionFromChunk.Config(
            question_client=client,
            question_temperature=1.0,
            answer_client=client,
            answer_temperature=0.0,
            answer_max_completion_tokens=384,
            chunker=SimpleCharacterChunker.Config(
                min_chunk_size_in_chars=500,
                max_chunk_size_in_chars=5_000,
            ),
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
            subcontext_generator=RandomizedSlidingWindowGenerator.Config(
                num_passes=2,
                min_window_size=3,
                max_window_size=4,
                min_step_size=1,
                max_step_size=2,
                seed=32,
            ),
        ),
        context=KalamangSectionedConfig(max_tokens_per_section=10_000),
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        num_samples=8_192,
        batch_size=256,
        max_num_batches_in_parallel=32,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
