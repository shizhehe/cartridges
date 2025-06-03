import os
from pathlib import Path
import pydrantic

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.subcontext import RandomSubcontextGenerator, RandomizedSlidingWindowGenerator
from capsules.tasks.swde import SWDEContextConfig
from capsules.utils import WandBConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-llama-3b-1xh100-min0-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

QUESTION_PROMPT_TEMPLATE = f"""Please generate a challenging question about the following excerpt from a website.

<excerpt>
{{chunk}}
</excerpt>

The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""


QUESTION_SYSTEM_PROMPT = f"""You are working to train a language model on a website.

You will be given excerpts of a website and your job is to generate a training question that tests reading comprehension of the website. 
After, we will train a langauge model to answer the question without access to the website (i.e. closed book). 
The question should be challenging and can require the model to remember specific details. 
"""


ANSWER_SYSTEM_PROMPT = f"""You are an english student. 
Please use the following information from a website to answer the question.

<excerpt>
{{subcontext}}
</excerpt>

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
    name=f"{file_name}_{html_page}",

    convo_generator=SimpleQuestionFromChunk.Config(
        question_client=client,
        question_temperature=0.9,
        question_max_completion_tokens=512,
        answer_client=client,
        answer_temperature=0.0,
        answer_max_completion_tokens=384,
        chunker=SimpleCharacterChunker.Config(
            min_chunk_size_in_chars=500,
            max_chunk_size_in_chars=10_000,
        ),
        question_prompt_template=QUESTION_PROMPT_TEMPLATE,
        answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
        question_system_prompt_template=QUESTION_SYSTEM_PROMPT,
        answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
        subcontext_generator=subcontext_generator,
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
        tags=[f"swde_imdb", "genbaseline", f"website_{html_page}"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
)

if __name__ == "__main__":
    pydrantic.main([config])


