import os
from pathlib import Path
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPromptWithChunk,
    QuestionSystemPrompt,
)
import pydrantic

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate.run import GenerateConfig
from capsules.generate.context_convo_generators.basic_question_from_chunk import (
    SimpleQuestionFromChunk,
)
from capsules.generate.chunk import DocumentChunker
from capsules.utils import WandBConfig
from capsules.tasks.enron import EnronContextConfig

client_config = TokasaurusClient.Config(
    #url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
    #url="https://hazyresearch--tokasaurus-llama-3b-serve.modal.run/v1",
    url="https://hazyresearch--tokasaurus-llama-3b-dp4-serve.modal.run/v1",
    model_name="not_empty"
)

CONTEXT_PROMPT = """Please generate a question about the following excerpt from an Enron email. 

{chunk}

The question should test a model's knowledge of the information in the email when asked in a closed book setting.
Generate only the question, with no other text or explanation."""

SYSTEM_PROMPT = (
    "You will be given excerpts from Enron emails and your job is to generate training questions that are grounded in the provided content. "
    "After we will train a language model to answer these questions without access to the original email (i.e. closed book). "
    "Therefore, the questions should be specific enough that the model can answer them without the context of the original email."
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
        chunker=DocumentChunker.Config(
            section_delimiter="\n\n",
            min_sections_per_chunk=1,
            max_documents_per_chunk=3,
        ),
        prompt_template=CONTEXT_PROMPT,
        question_system_prompt_generator=QuestionSystemPrompt.Config(
            prompt_template=SYSTEM_PROMPT
        ),
        answer_system_prompt_generator=AnswerSystemPromptWithChunk.Config(),
    ),
    context=EnronContextConfig(
        split="train",
        max_samples=32_768,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32_768,
    batch_size=256,
    max_num_batches_in_parallel=8,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)

if __name__ == "__main__":
    pydrantic.main([config]) 