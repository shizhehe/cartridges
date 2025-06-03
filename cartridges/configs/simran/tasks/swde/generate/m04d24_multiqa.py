# goal is to have the model generate multiple QA pairs towards better diversity"

import os
from pathlib import Path
import pydrantic
from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.configs.simran.tasks.swde.generators.m04d21_simpleqa import SimpleQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.subcontext import RandomSubcontextGenerator
from capsules.tasks.swde import SWDEContextConfig
from capsules.utils import WandBConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max24-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

QUESTION_PROMPT_TEMPLATE = f"""Please generate a non multiple choice question to test knowledge of the movie attribute details that are mentioned in the passage. Focus your questions on numbers, entities, dates, lists, groups of people, quotes, ratings, and other movie attributes.

<passage>
{{chunk}}
</passage>

The question should test knowledge of the IMDB movie attributes. Use the following format:

<question> your question here </question>
"""


QUESTION_SYSTEM_PROMPT = f""""""


ANSWER_SYSTEM_PROMPT = f"""Please use the following passage from the movie website to answer the question.

<passage>
{{subcontext}}
</passage>
"""

ANSWER_PROMPT_TEMPLATE = """Here's a question that tests a model's knowledge of the information in the passage: {question}

Please provide the answer using the following tag format:

<start> answer </end>"""


num_samples = 32768*2

file_name = Path(__file__).stem

subcontext_generator = RandomSubcontextGenerator.Config(
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    min_num_chunks=1,
    max_num_chunks=1,
    num_contexts=100,
    seed=32,
)

html_page = '0349.htm'

configs = []
for html_page in ['0000.htm']: 
    config = GenerateTrainingConfig(
        name=f"{file_name}_{html_page}_temp0.3",

        convo_generator=SimpleQuestionFromChunk.Config(
            question_client=client,
            question_temperature=0.9,
            question_max_completion_tokens=512,
            answer_client=client,
            answer_temperature=0.3,
            answer_max_completion_tokens=384,
            chunker=SimpleCharacterChunker.Config(
                min_chunk_size_in_chars=1_000,
                max_chunk_size_in_chars=15_000,
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

        save_wandb_artifact=False,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"swde_imdb", "genbaseline", f"website_{html_page}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)




