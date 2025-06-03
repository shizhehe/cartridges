import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.configs.ryan.m03d27.doc_names import DOC_NAMES


from capsules.data.mtob import KalamangSectionedConfig
from capsules.generate.generators.memorization import LocateFairSectionGenerator
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

file_name = Path(__file__).stem


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(4)],
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)


def config():
    return GenerateConfig(
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=LocateFairSectionGenerator.Config(
            answer_client=client,
            answer_temperature=0.0,
            answer_max_completion_tokens=384,
            chunk_size_range=(500, 2000),
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
            subcontext_generator=RandomSubcontextGenerator.Config(
                num_contexts=120,
                min_num_chunks=3,
                max_num_chunks=4,
                seed=32,
            ),
        ),
        context=KalamangSectionedConfig(max_tokens_per_section=10_000, book_size="medium"),
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        num_samples=8_192,
        batch_size=128,
        max_num_batches_in_parallel=64,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
