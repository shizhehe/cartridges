import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable


from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.clients.openai_batch import TokasaurusClient

from capsules.configs.ryan.m03d27.doc_names import DOC_NAMES


from capsules.generate.run import GenerateConfig
from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.subcontext import SequentialSubcontextGenerator, RandomizedSlidingWindowGenerator
from capsules.tasks.finance import FinanceBenchContextConfig
from capsules.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    # url="https://hazyresearch--tokasaurus-llama-3b-8xh100-min0-serve.modal.run/v1", 
    url="http://localhost:8889",
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)

client = TokasaurusClient.Config(
    url="http://localhost:8889/v1",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

QUESTION_PROMPT_TEMPLATE = """Please generate a challenging question about the following excerpt from a document. 

---

{chunk}

---
The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""

QUESTION_SYSTEM_PROMPT = f"""You are working to train a language model on the information in a document.

You will be given excerpts of the document and your job is to generate a training question that is grounded in the provided document. 
After we will train a langauge model to answer the question without access to the document (i.e. closed book). 
The question should be challenging and can require the model to remember details like specific numerical figures, dates, and names. 

Here is the content of the document.
{{subcontext}}
"""


ANSWER_SYSTEM_PROMPT = f"""You are an expert financial analyst. 
Please use the following information from a financial document to answer the question.

Here is the content of the document.
{{subcontext}}

First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""

SUBCONTEXT_TYPE = os.environ.get("SUBCONTEXT_TYPE", "sequential")

if SUBCONTEXT_TYPE == "sequential":
    subcontext_generator = SequentialSubcontextGenerator.Config(
        num_passes_over_document=1,
        min_length_in_chars=8192,
        max_length_in_chars=16384,
    )
elif SUBCONTEXT_TYPE == "random_sliding":
    subcontext_generator = RandomizedSlidingWindowGenerator.Config(
        num_passes=10,
        min_window_size=2,
        max_window_size=4,
        min_step_size=3,
        max_step_size=5,
        seed=32,
    )
else:
    raise ValueError(f"Invalid subcontext type: {SUBCONTEXT_TYPE}")

ANSWER_PROMPT_TEMPLATE = """{question}"""
generator_config = SimpleQuestionFromChunk.Config(
    question_client=client,
    question_temperature=0.9,
    question_max_completion_tokens=512,
    answer_client=client,
    answer_temperature=0.0,
    answer_max_completion_tokens=384,
    chunker=SimpleCharacterChunker.Config(
        min_chunk_size_in_chars=500,
        max_chunk_size_in_chars=10_000,
    ),
    question_prompt_template=QUESTION_PROMPT_TEMPLATE,
    answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
    question_system_prompt_template=QUESTION_SYSTEM_PROMPT,
    answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
    subcontext_generator=subcontext_generator,
)

file_name = Path(__file__).stem


DOC_NAME = "AMD_2022_10K"

config = GenerateConfig(
    name=FormatStringVariable(f"{file_name}_{SUBCONTEXT_TYPE}_{DOC_NAME}_n{{num_samples}}"),
    convo_generator=generator_config,
    context=FinanceBenchContextConfig(doc_names=[DOC_NAME], split_on="page"),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=256,
    max_num_batches_in_parallel=16,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    pydrantic.main(config)
