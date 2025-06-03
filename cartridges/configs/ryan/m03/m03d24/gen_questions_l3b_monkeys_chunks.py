import os
from pathlib import Path
from capsules.data.paths import MONKEYS, MONKEYS_TITLE

from capsules.generate.context_convo_generators.school.teaching_v2_no_cot import AnswerSystemPromptForTeaching, QuestionFromDocAndChunk, QuestionSystemPromptForTeaching

import pydrantic


from capsules.clients.tokasaurus import TokasaurusClient

from capsules.generate.run import GenerateConfig
from capsules.generate.context_convo_generators.basic_question_from_chunk import (
    SimpleQuestionFromChunk,
)
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.utils import WandBConfig

question_client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)
answer_client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)



file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=QuestionFromDocAndChunk.Config(
        question_client=question_client_config,
        question_temperature=0.6,
        question_max_completion_tokens=512,
        answer_client=answer_client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=384,
        chunker=SimpleCharacterChunker.Config(
            min_chunk_size_in_chars=500,
            max_chunk_size_in_chars=10_000,
        ),
        question_system_prompt_generator=QuestionSystemPromptForTeaching.Config(),
        answer_system_prompt_generator=AnswerSystemPromptForTeaching.Config(
            include_chunk=False
        ),
    ),
    document_title=MONKEYS_TITLE,
    document_path_or_url=str(MONKEYS.absolute()),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=12_000,
    batch_size=32,
    max_num_batches_in_parallel=24,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
