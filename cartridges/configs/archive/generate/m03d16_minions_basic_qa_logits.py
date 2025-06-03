import os
from pathlib import Path
from capsules.generate.context_convo_generators.questions_from_files import (
    ConvoGeneratorWithLLMAnswerQuestionsFromFile,
)
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
    url="http://localhost:8012/v1", model_name="not_empty"
)


file_name = Path(__file__).stem

config = GenerateConfig(
    name=file_name,
    convo_generator=SimpleQuestionFromChunk.Config(
        question_client=client_config,
        question_temperature=0.6,
        question_max_completion_tokens=256,
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=256,
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
    num_samples=64_000,
    batch_size=128,
    max_num_batches_in_parallel=20,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
    previous_run_dir=Path("/scr/ryanehrlich/capsules/data_dir/2025-03-16-23-39-01-m03d16_minions_basic_qa_logits/d26cf36e-87ea-4b36-a726-e4d922ea968b")
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
