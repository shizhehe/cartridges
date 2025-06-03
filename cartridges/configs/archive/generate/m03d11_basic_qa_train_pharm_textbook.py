import os
from pathlib import Path
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithChunk,
    AnswerSystemPromptWithDoc,
    QuestionSystemPromptWithDoc,
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


# client_config = TokasaurusClient.Config(
#     url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
#     model_name="not_empty"
# )


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
        answer_max_completion_tokens=512,
        chunker=SimpleCharacterChunker.Config(
            min_chunk_size_in_chars=500,
            max_chunk_size_in_chars=25_00,
        ),
        question_system_prompt_generator=QuestionSystemPromptWithDoc.Config(),
        answer_system_prompt_generator=AnswerSystemPromptWithDoc.Config(),
    ),
    document_title="Pharmacology for Nurses",
    document_path_or_url=str(
        (
            Path(__file__).parent.parent.parent.parent / "data/openstax_textbooks/pharm"
        ).absolute()
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=48_000,#64_000,
    num_top_logprobs=20,
    batch_size=128,
    max_num_batches_in_parallel=20,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
    previous_run_dir=Path("/scr/ryanehrlich/capsules/data_dir/2025-03-11-16-49-52-m03d11_basic_qa_train_pharm_textbook/2b4c9b7f-f046-4f87-9318-35fcddc20897"),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
