import os
from pathlib import Path
from cartridges.generate.context_convo_generators.system_prompts import (
    AnswerSystemPrompt,
    AnswerSystemPromptWithChunk,
    AnswerSystemPromptWithDoc,
    QuestionSystemPrompt,
    QuestionSystemPromptWithDoc,
    QuestionSystemPromptWithTitle,
)
import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.together import TogetherClient
from cartridges.clients.openai import OpenAIClient
from cartridges.generate.run import GenerateConfig
from cartridges.generate.context_convo_generators.basic_question_from_chunk import (
    SimpleQuestionFromChunk,
)
from cartridges.generate.chunk import DocumentChunker
from cartridges.utils import WandBConfig
from cartridges.tasks.longhealth import LongHealthContextConfig

client_config = TokasaurusClient.Config(
    # url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
    url="https://hazyresearch--tokasaurus-llama-3b-dp4-serve.modal.run/v1",
    model_name="llama-3.2-3b-instruct"  # this argument does not matter
)


QUESTION_PROMPT_TEMPLATE = """Please generate a multiple choicequestion about the following excerpt of text from the document. 

{chunk}

The question should test a model's knowledge of the information in the document when asked in a closed book setting. 
Generate only the question (between <question> and </question> tags) and the options (between <options> and </options> tags) with no other text or explanation."""

QUESTION_SYSTEM_PROMPT = """
We want to train a language model on the information in a patient's medical records.
Below is the patient's medical records between <records> and </records> tags.

You will be given excerpts of the medical records and your job is to generate training questions that are grounded in the provided patient medical records. 
After we will train a langauge model to answer these questions without access to the medical records (i.e. closed book). 

Here are some examples of questions that you could generate (these are for a different patient):

<examples>
    <question>
    On what date was a CT scan of the chest/abdomen/pelvis with contrast agent performed in Laura Miller?
    </question>

    <options>
    May 28th 2020
    June 13th 2020
    June 18th 2020
    June 15th 2020
    June 19th 2020
    </options>
</example>

Below is the patient's medical records between <records> and </records> tags.

<records>
{context}
</records>
"""


ANSWER_SYSTEM_PROMPT = """You are a doctor. 
Please use the following information from the patient's medical records to answer the question.

---

{context}"""


ANSWER_PROMPT_TEMPLATE = """
Please answer the question below: 

{question}

You should first think step by step. Then give your final answer exactly as it appears in the options with the following format:

<answer>
{{answer}}
</answer>"""

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
        question_prompt_template=QUESTION_PROMPT_TEMPLATE,
        answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
        question_system_prompt_generator=QuestionSystemPrompt.Config(
            prompt_template=QUESTION_SYSTEM_PROMPT
        ),
        answer_system_prompt_generator=AnswerSystemPrompt.Config(
            prompt_template=ANSWER_SYSTEM_PROMPT
        ),
        num_top_logprobs=20
    ),
    context=LongHealthContextConfig(
        patient_ids=["patient_05"]
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    # generate config
    num_samples=8192,
    # num_samples=128,
    # num_top_logprobs=20,
    batch_size=256,
    max_num_batches_in_parallel=8,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
    ),
    save_wandb_preview=True
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
