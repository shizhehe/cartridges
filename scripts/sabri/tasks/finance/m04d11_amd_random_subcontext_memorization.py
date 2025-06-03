import os
from pathlib import Path


import pydrantic

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.configs.ryan.m03d27.doc_names import DOC_NAMES


from cartridges.generate.generators.memorization import LocateFairSectionGenerator
from cartridges.generate.generate_training import GenerateTrainingConfig

from pydrantic.variables import FormatStringVariable


from cartridges.generate.subcontext import (
    RandomSubcontextGenerator,
    RandomizedSlidingWindowGenerator,
)

from cartridges.tasks.finance import FinanceBenchContextConfig
from cartridges.utils import WandBConfig

ANSWER_SYSTEM_PROMPT = """You are an expert financial analyst. 
Please use the following information from a financial document to answer the question.


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
    ports=[8880 + i for i in range(8)],
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


def config():
    return GenerateTrainingConfig(
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=LocateFairSectionGenerator.Config(
            answer_client=client,
            answer_temperature=0.0,
            answer_max_completion_tokens=384,
            chunk_size_range=(500, 2000),
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
            subcontext_generator=RandomSubcontextGenerator.Config(
                tokenizer="meta-llama/Llama-3.2-3B-Instruct",
                max_tokens_per_section=2_000,
                num_contexts=100,
                min_num_chunks=6,
                max_num_chunks=8,
                seed=32,
            ),
        ),
        context=FinanceBenchContextConfig(doc_name="AMD_2022_10K", split_on="page"),
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        num_samples=16_384,
        batch_size=512,
        max_num_batches_in_parallel=32,
        wandb=WandBConfig(
            project="cartridges",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
