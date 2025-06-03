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
from cartridges.generate.chunk import DocumentChunker, LeveledDocumentChunker
from cartridges.tasks.finance import FinanceBenchContextConfig
from cartridges.utils import WandBConfig
from cartridges.tasks.longhealth import LongHealthContextConfig

client_config = TokasaurusClient.Config(
    # url="https://scalingintelligence--tokasaurus-llama-3b-serve.modal.run/v1",
    # url="https://hazyresearch--tokasaurus-llama-3b-dp4-serve.modal.run/v1",
    url="https://hazyresearch--tokasaurus-llama-3b-8xh100-serve.modal.run/v1",
    model_name="llama-3.2-3b-instruct"  # this argument does not matter
)


QUESTION_PROMPT_TEMPLATE = """Please generate a challenging question about the following excerpt from a financial document. 

---

{chunk}

---
The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""

QUESTION_SYSTEM_PROMPT = """You are working to train a language model on the information in a financial document.

You will be given excerpts of the financial document and your job is to generate a training question that is grounded in the provided financial document. 
After we will train a langauge model to answer the question without access to the financial document (i.e. closed book). 
The question should be challenging and can require the model to remember details like specific numerical figures, dates, and names. 

Below is the title of the financial document.
<title>
{title}
</title>
"""


ANSWER_SYSTEM_PROMPT = """You are an expert financial analyst. 
Please use the following information from a financial document to answer the question.

<excerpt>
{chunk}
</excerpt>

First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""


ANSWER_PROMPT_TEMPLATE = """{question}"""

# DOC_NAME = "3M_2018_10K"
# DOC_NAME = "BOEING_2022_10K"
# DOC_NAME = "AMERICANEXPRESS_2022_10K"
DOC_NAME = "AMD_2022_10K"

file_name = Path(__file__).stem

config = GenerateConfig(
    name=f"{file_name}_{DOC_NAME}",
    convo_generator=SimpleQuestionFromChunk.Config(
        question_client=client_config,
        question_temperature=0.6,
        question_max_completion_tokens=256,
        answer_client=client_config,
        answer_temperature=0.0,
        answer_max_completion_tokens=512,
        chunker=LeveledDocumentChunker.Config(
            level_to_weight={
                0: 0, 
                1: 0.0,
                2: 0.5,
                3: 0.6,
                4: 5.0,
            },
            max_tokens=12_000,
            tokenizer="meta-llama/Llama-3.2-1B-Instruct",
        ),  # one document per chunk
        question_prompt_template=QUESTION_PROMPT_TEMPLATE,
        answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
        question_system_prompt_generator=QuestionSystemPrompt.Config(
            prompt_template=QUESTION_SYSTEM_PROMPT,
            max_tokens=12_000,
            tokenizer="meta-llama/Llama-3.2-1B-Instruct",
        ),
        answer_system_prompt_generator=AnswerSystemPrompt.Config(
            prompt_template=ANSWER_SYSTEM_PROMPT,
            max_tokens=12_000,
            tokenizer="meta-llama/Llama-3.2-1B-Instruct",
        ),
        num_top_logprobs=20
    ),
    context=FinanceBenchContextConfig(
        doc_names=[DOC_NAME]
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=128,
    max_num_batches_in_parallel=16,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
    ),

    save_wandb_preview=True,
    previous_run_dir=Path(
        "/data/sabri/code/Cartridges/output/2025-03-25-16-35-28-m03d22_generate_finance_single/8d000973-7369-44fd-89cf-b50ee47edbab"
    )
)


if __name__ == "__main__":
    # Launch pydrantic CLI, which will parse arguments and run config.run() if desired.
    pydrantic.main([config])
