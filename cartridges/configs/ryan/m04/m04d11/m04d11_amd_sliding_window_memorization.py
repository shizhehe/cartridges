import os
from pathlib import Path


import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.configs.ryan.m03d27.doc_names import DOC_NAMES


from capsules.generate.generators.memorization import LocateFairSectionGenerator
from capsules.generate.run_new import GenerateConfig

from pydrantic.variables import FormatStringVariable


from capsules.generate.subcontext import (
    RandomizedSlidingWindowGenerator,
)

from capsules.tasks.finance import FinanceBenchContextConfig, FinanceBenchSectionedContextConfig
from capsules.utils import WandBConfig

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
    return GenerateConfig(
        name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
        convo_generator=LocateFairSectionGenerator.Config(
            answer_client=client,
            answer_temperature=0.0,
            answer_max_completion_tokens=384,
            chunk_size_range=(500, 2000),
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
            subcontext_generator=RandomizedSlidingWindowGenerator.Config(
                num_passes=5,
                min_window_size=6,
                max_window_size=8,
                min_step_size=2,
                max_step_size=5,
                seed=32,
            ),
        ),
        context=FinanceBenchSectionedContextConfig(doc_name="AMD_2022_10K", max_tokens_per_section=2_000),
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        num_samples=32768,
        batch_size=512,
        max_num_batches_in_parallel=32,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    pydrantic.main([config()])
