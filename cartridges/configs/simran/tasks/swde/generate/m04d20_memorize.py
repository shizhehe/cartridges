import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.subcontext import RandomSubcontextGenerator
from capsules.generate.generators.memorization import LocateFairSectionGenerator
from capsules.tasks.swde import SWDEContextConfig
from capsules.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-llama-3b-1xh100-min0-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


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

num_samples = 32768

file_name = Path(__file__).stem

subcontext_generator = RandomSubcontextGenerator.Config(
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    min_num_chunks=1,
    max_num_chunks=4,
    num_contexts=16384,
    seed=32,
)


html_page = '0349.htm'
config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{html_page}_memorize"),
    
    convo_generator=LocateFairSectionGenerator.Config(
        answer_client=client,
        answer_temperature=0.0,
        answer_max_completion_tokens=384,
        chunk_size_range=(500, 2000),
        answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
        answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
        subcontext_generator=subcontext_generator
    ),

    context=SWDEContextConfig(
        webpage_id=html_page,
        max_tokens_per_section=10_000,
    ),
    
    # generate config
    num_samples=num_samples,
    batch_size=min(num_samples, 256),
    max_num_batches_in_parallel=min(num_samples, 32),

    wandb=WandBConfig(
        project="capsules",
        tags=[f"swde_imdb", "genbaseline", f"webpage_{html_page}"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
)

if __name__ == "__main__":
    pydrantic.main([config])


