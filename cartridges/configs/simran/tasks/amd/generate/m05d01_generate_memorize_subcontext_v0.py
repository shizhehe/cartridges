import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.generate.generators.memorization import LocateFairSectionGenerator
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.subcontext import (RandomSubcontextGenerator)
from capsules.tasks.finance import FinanceBenchSectionedContextConfig
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
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


parallel = True
num_samples = 32678 if parallel else 1
batch_size = min(512, num_samples) if parallel else 1
parallel_batches = min(32, num_samples) if parallel else 1


memorize_config =  GenerateTrainingConfig(

    name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),

    convo_generator=LocateFairSectionGenerator.Config(
        answer_client=client,
        answer_temperature=0.0,
        answer_max_completion_tokens=384,
        chunk_size_range=(500, 2000),
        answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
        answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
        subcontext_generator=RandomSubcontextGenerator.Config(
            num_contexts=100,
            min_num_chunks=6,
            max_num_chunks=8,
            seed=32,
        ),
    ),
    
    context=FinanceBenchSectionedContextConfig(
        doc_name="AMD_2022_10K", 
        max_tokens_per_section=2_000
    ),

    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=num_samples,
    batch_size=batch_size,
    max_num_batches_in_parallel=parallel_batches,
    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)

configs = []
configs.append(memorize_config)


if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])


