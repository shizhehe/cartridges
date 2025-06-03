import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient

from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.subcontext import RandomSubcontextGenerator, RandomizedSlidingWindowGenerator
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(8)],
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


STATE = "Alabama"


QUESTION_PROMPT_TEMPLATE = f"""Please generate a challenging question about the following excerpt from the state legal statutes of {STATE}. 

---

{{chunk}}

---
The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""

QUESTION_SYSTEM_PROMPT = f"""You are working to train a language model on the {STATE} state legal statutes.

You will be given excerpts of {STATE} state statutes and your job is to generate a training question that is grounded in the provided statutes. 
After, we will train a langauge model to answer the question without access to the statute text (i.e. closed book). 
The question should be challenging and can require the model to remember specific legal details. 
"""


ANSWER_SYSTEM_PROMPT = f"""You are a legal expert on the state statutes of {STATE}. 
Please use the following information from the {STATE} state statutes to answer the question.

<excerpt>
{{subcontext}}
</excerpt>

First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""

file_name = Path(__file__).stem

STATE = "Alabama"

subcontext_generator = RandomSubcontextGenerator.Config(
    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    min_num_chunks=1,
    max_num_chunks=6,
    num_contexts=16384,
    seed=32,
)

config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{STATE.lower()}_n{{num_samples}}_nstatutes{{context.num_statutes}}"),
    convo_generator=SimpleQuestionFromChunk.Config(
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
    ),
    context=HousingStatutesContextConfig(
        states=[STATE],
        split_on="statute",
        num_statutes=100
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=256,
    batch_size=256,
    max_num_batches_in_parallel=32,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
