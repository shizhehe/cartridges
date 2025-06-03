import os
from pathlib import Path
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithChunk,
    QuestionSystemPromptWithEntireContext,
)
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.clients.together import TogetherClient
from capsules.clients.openai import OpenAIClient
from capsules.generate.run import GenerateConfig
from capsules.generate.context_convo_generators.basic_question_from_chunk import (
    SimpleQuestionFromChunk,
)
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.utils import WandBConfig


client_config = TokasaurusClient.Config(
    url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
    model_name="not_empty"
)


# client_config = TokasaurusClient.Config(
#     url="http://localhost:8012/v1", model_name="not_empty"
# )


file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=SimpleQuestionFromChunk.Config(
        question_client=client_config,
        question_temperature=0.6,
        question_max_completion_tokens=256,
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=1024,
        chunker=SimpleCharacterChunker.Config(
            min_chunk_size_in_chars=500,
            max_chunk_size_in_chars=10_000,
        ),
        question_system_prompt_generator=QuestionSystemPromptWithEntireContext.Config(),
        answer_system_prompt_generator=AnswerSystemPromptWithChunk.Config(),
    ),
    document_title="Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models",
    document_path_or_url=str(
        (
            Path(__file__).parent.parent.parent.parent / "data/example_docs/minions.txt"
        ).absolute()
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=128,
    num_top_logprobs=20,
    batch_size=128,
    max_num_batches_in_parallel=10,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
