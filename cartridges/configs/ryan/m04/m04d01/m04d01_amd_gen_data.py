import os
from pathlib import Path


import pydrantic


from capsules.clients.tokasaurus import TokasaurusClient

from capsules.configs.ryan.m03d27.doc_names import DOC_NAMES
from capsules.generate.context_convo_generators.system_prompts import (
    AnswerSystemPrompt,
    QuestionSystemPrompt,
)
from capsules.generate.run import GenerateConfig
from capsules.generate.context_convo_generators.basic_question_from_chunk import (
    SimpleQuestionFromChunk,
)
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.tasks.finance import FinanceBenchContextConfig
from capsules.utils import WandBConfig

question_client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)
answer_client_config = TokasaurusClient.Config(
    url="http://localhost:8012/v1", model_name="not_empty"
)


QUESTION_PROMPT_TEMPLATE = """Please generate a challenging question about the following excerpt from a document. 

---

{chunk}

---
The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""

QUESTION_SYSTEM_PROMPT = """You are working to train a language model on the information in a document.

You will be given excerpts of the document and your job is to generate a training question that is grounded in the provided document. 
After we will train a langauge model to answer the question without access to the document (i.e. closed book). 
The question should be challenging and can require the model to remember details like specific numerical figures, dates, and names. 

Below is the title of the document.
<title>
{title}
</title>

Here is the content of the document.
<document>
{context}
</document>
"""


ANSWER_SYSTEM_PROMPT = """Please use the information in the following document to answer the user's questions.
<title>
{title}
</title>

<document>
{context}
</document>

Here is a relevant excerpt.

<excerpt>
{chunk}
</excerpt>

First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""

file_name = Path(__file__).stem


def config(doc_name):
    assert doc_name in DOC_NAMES

    return GenerateConfig(
        name=f"{file_name}_3b",
        convo_generator=SimpleQuestionFromChunk.Config(
            question_client=question_client_config,
            question_temperature=0.9,
            question_max_completion_tokens=512,
            answer_client=answer_client_config,
            answer_temperature=0.0,
            answer_max_completion_tokens=384,
            chunker=SimpleCharacterChunker.Config(
                min_chunk_size_in_chars=500,
                max_chunk_size_in_chars=10_000,
            ),
            question_prompt_template=QUESTION_PROMPT_TEMPLATE,
            answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
            question_system_prompt_generator=QuestionSystemPrompt.Config(
                prompt_template=QUESTION_SYSTEM_PROMPT,
            ),
            answer_system_prompt_generator=AnswerSystemPrompt.Config(
                prompt_template=ANSWER_SYSTEM_PROMPT,
            ),
        ),
        context=FinanceBenchContextConfig(doc_names=[doc_name], force_single_doc=True),
        output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
        # generate config
        num_samples=64_000,
        batch_size=512,
        max_num_batches_in_parallel=8,
        wandb=WandBConfig(
            project="capsules",
            entity="hazy-research",
        ),
    )


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    doc_name = "AMD_2022_10K"
    pydrantic.main(config(doc_name))
