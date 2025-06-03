import os
from pathlib import Path
from capsules.generate.context_convo_generators.system_prompts import AnswerSystemPromptWithChunk, QuestionSystemPromptWithEntireContext
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.clients.together import TogetherClient
from capsules.clients.openai import OpenAIClient
from capsules.generate.run import GenerateConfig
from capsules.generate.context_convo_generators.creative_question_from_chunk import CreativeQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.utils import WandBConfig


# client_config = TokasaurusClient.Config(
#     url="http://localhost:8012/v1",
#     model_name="not_empty"
# )

client_config = TogetherClient.Config(
    model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    
)

# client_config = OpenAIClient.Config(
#     model_name="gpt-4o-mini",
# )


file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=CreativeQuestionFromChunk.Config(
        question_client=client_config,
        question_temperature=0.6,
        question_max_completion_tokens=256,

        num_top_logprobs = 1,

        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=1024,

        question_system_prompt_generator=QuestionSystemPromptWithEntireContext.Config(),
        answer_system_prompt_generator=AnswerSystemPromptWithChunk.Config(),

        chunker=SimpleCharacterChunker.Config(
            min_chunk_size_in_chars=500,
            max_chunk_size_in_chars=10_000,
        ),
    ),

    document_title="Full Bee Movie Script",
    document_path_or_url=str(
        (
            Path(__file__).parent.parent.parent.parent / "data/example_docs/bee_movie_script.txt"
        ).absolute()
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # generate config
    num_samples=4,
    num_top_logprobs=1,
    batch_size=1024,
    max_num_batches_in_parallel=1,
    
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
