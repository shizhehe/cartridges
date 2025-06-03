import os
from pathlib import Path
from capsules.generate.context_convo_generators.questions_from_files import (
    ConvoGeneratorWithLLMAnswerQuestionsFromFile,
)
from capsules.data.paths import MINIONS
from capsules.generate.context_convo_generators.school.evaluation import AnswerSystemPromptForTesting, EvaluationFromDoc, EvaluationFromDocAndChunk, QuestionSystemPromptForTesting

import pydrantic

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

file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=EvaluationFromDocAndChunk.Config(
        question_client=client_config,
        question_temperature=1.0,
        question_max_completion_tokens=256,
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=256,
        question_system_prompt_generator=QuestionSystemPromptForTesting.Config(),
        answer_system_prompt_generator=AnswerSystemPromptForTesting.Config(),
        chunker=SimpleCharacterChunker.Config(
            min_chunk_size_in_chars=500,
            max_chunk_size_in_chars=10_000,
        ),
    ),
    document_title="Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models",
    document_path_or_url=str(MINIONS.absolute()),
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
