import os
from pathlib import Path
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithChunk,
    AnswerSystemPromptWithDoc,
    QuestionSystemPromptWithDoc,
    QuestionSystemPromptWithTitle,
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
from capsules.generate.chunk import DocumentChunker
from capsules.utils import WandBConfig
from capsules.tasks.longhealth import LongHealthContextConfig

client_config = TokasaurusClient.Config(
    # url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
    url="https://hazyresearch--tokasaurus-llama-3b-serve-dev.modal.run/v1",
    model_name="not_empty"
)


# client_config = TokasaurusClient.Config(
#     url="http://localhost:8012/v1", model_name="not_empty"
# )


file_name = Path(__file__).stem

patient_ids = ["patient_01"]

config = GenerateConfig(
    name=file_name,
    convo_generator=SimpleQuestionFromChunk.Config(
        question_client=client_config,
        question_temperature=0.6,
        question_max_completion_tokens=256,
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=512,
        chunker=DocumentChunker.Config(
            section_delimiter="\n\n",
            min_sections_per_chunk=4,
            max_documents_per_chunk=3,
        ),
        # need to used title, because context is otherwise too long
        question_system_prompt_generator=QuestionSystemPromptWithTitle.Config(),
        answer_system_prompt_generator=AnswerSystemPromptWithChunk.Config(),
    ),
    context=LongHealthContextConfig(
        patient_ids=patient_ids
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=32_768,
    num_top_logprobs=20,
    batch_size=256,
    max_num_batches_in_parallel=8,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
