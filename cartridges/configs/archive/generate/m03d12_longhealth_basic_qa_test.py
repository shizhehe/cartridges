import os
from pathlib import Path
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithChunk,
    AnswerSystemPromptWithDoc,
    QuestionSystemPrompt,
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

from capsules.configs.generate.m03d12_longhealth_basic_qa_train import CONTEXT_PROMPT, SYSTEM_PROMPT


client_config = OpenAIClient.Config(model_name="gpt-4o")

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
        chunker=DocumentChunker.Config(
            section_delimiter="\n\n",
            min_sections_per_chunk=4,
            max_documents_per_chunk=3,
        ),
        prompt_template=CONTEXT_PROMPT,
        question_system_prompt_generator=QuestionSystemPrompt.Config(
            prompt_template=SYSTEM_PROMPT
        ),
        answer_system_prompt_generator=AnswerSystemPromptWithChunk.Config(),
    ),
    context=LongHealthContextConfig(patient_ids=None), # Include all patients
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=256,
    # num_samples=128,
    # num_top_logprobs=20,
    batch_size=4,
    max_num_batches_in_parallel=32,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
