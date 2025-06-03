import os
from pathlib import Path
from capsules.generate.context_convo_generators.questions_from_files import (
    ConvoGeneratorWithLLMAnswerQuestionsFromFile,
)
from capsules.data.paths import AZALIA_FAST, AZALIA_FAST_TITLE, MINIONS, MONKEYS
from capsules.generate.context_convo_generators.school.evaluation import AnswerSystemPromptForTesting, EvaluationFromDoc, QuestionSystemPromptForTesting
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

openai_client_config = OpenAIClient.Config(
    model_name="gpt-4o",
)

toka_client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)


file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=EvaluationFromDoc.Config(
        question_client=openai_client_config,
        question_temperature=1.0,
        question_max_completion_tokens=256,
        answer_client=toka_client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=384,
        question_system_prompt_generator=QuestionSystemPromptForTesting.Config(),
        answer_system_prompt_generator=AnswerSystemPromptForTesting.Config(),
    ),
    document_title=AZALIA_FAST_TITLE,
    document_path_or_url=str(AZALIA_FAST.absolute()),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=128,
    batch_size=4,
    max_num_batches_in_parallel=10,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
