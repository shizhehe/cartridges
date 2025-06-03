import os
from pathlib import Path
import pydrantic

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.configs.simran.tasks.swde.generators.m04d21_simpleqa import SimpleQuestionFromChunk
from cartridges.generate.chunk import SimpleCharacterChunker
from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.subcontext import RandomSubcontextGenerator
from cartridges.tasks.swde import SWDEContextConfig
from cartridges.utils import WandBConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max24-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


QUESTION_PROMPT_TEMPLATE = f"""We are going to be generating a novel QA pair from the website passage. We will then test a student on whether they know details about the movie. Please generate a non multiple choice question.

<passage>
{{chunk}}
</passage>

The question should test a model's knowledge of the information in the passage. Put the question in the following tag format:

<start> your question </end>
"""

QUESTION_SYSTEM_PROMPT = f""""""


ANSWER_SYSTEM_PROMPT = f"""Please use the following passage from the movie website to complete the Question-Answer pair.

<passage>
{{subcontext}}
</passage>

Directly copy the answer form the website without adding extra verbiage.
"""


ANSWER_PROMPT_TEMPLATE = """
Question: {question}
Answer:"""


num_samples = 32768

file_name = Path(__file__).stem

subcontext_generator = RandomSubcontextGenerator.Config(
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    min_num_chunks=1,
    max_num_chunks=3,
    num_contexts=100,
    seed=32,
)


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
                min_chunk_size_in_chars=3_000,
                max_chunk_size_in_chars=6_000,
            ),
            question_prompt_template=QUESTION_PROMPT_TEMPLATE,
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            question_system_prompt_template=QUESTION_SYSTEM_PROMPT,
            answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
            subcontext_generator=subcontext_generator,
        ),

        context=SWDEContextConfig(
            webpage_id=html_page,
            max_tokens_per_section=-1,
            pages_path="/data/sabri/data/evaporate/swde/movie/movie-imdb(2000)",
            table_path="/data/sabri/data/evaporate/data/swde_movie_imdb/table.json"
        ),
        
        # generate config
        num_samples=num_samples,
        batch_size=min(num_samples, 256),
        max_num_batches_in_parallel=min(num_samples, 1),

        save_wandb_artifact=False,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"swde_imdb", "genbaseline", f"website_{html_page}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)



