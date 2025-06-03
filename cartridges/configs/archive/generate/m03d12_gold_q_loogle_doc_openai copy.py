import os
from pathlib import Path
from capsules.generate.context_convo_generators.questions_from_files import (
    ConvoGeneratorWithLLMAnswerQuestionsFromFile,
)
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithChunk,
    AnswerSystemPromptWithEntireContext,
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

from capsules.clients.openai import OpenAIClient

client_config = OpenAIClient.Config(
    model_name="gpt-4o",
)


# client_config = TokasaurusClient.Config(
#     url="http://localhost:8012/v1", model_name="not_empty"
# )


file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=ConvoGeneratorWithLLMAnswerQuestionsFromFile.Config(
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=512,
        answer_system_prompt_generator=AnswerSystemPromptWithEntireContext.Config(),
        file=(
            Path(__file__).parent.parent.parent.parent
            / "data/loogle/cc4f70a5_questions.txt"
        ).absolute(),
    ),
    document_title="57th Medical Detachment",
    document_path_or_url=str(
        (
            Path(__file__).parent.parent.parent.parent / "data/loogle/cc4f70a5.txt"
        ).absolute(),
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=56, # num questions for this specific doc
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
