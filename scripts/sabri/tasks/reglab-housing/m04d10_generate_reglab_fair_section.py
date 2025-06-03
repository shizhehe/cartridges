import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable


from cartridges.clients.openai_batch import TokasaurusClient
from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.generators.memorization import LocateFairSectionGenerator
from cartridges.generate.subcontext import RandomSubcontextGenerator, RandomizedSlidingWindowGenerator
from cartridges.tasks.reglab.housing_qa import HousingStatutesContextConfig
from cartridges.utils import WandBConfig


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(8)],
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

SUBCONTEXT_TYPE = os.environ.get("SUBCONTEXT_TYPE", "random")

if SUBCONTEXT_TYPE == "random":
    subcontext_generator = RandomSubcontextGenerator.Config(
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        min_num_chunks=1,
        max_num_chunks=6,
        num_contexts=16384,
        seed=32,
    )
else:
    raise ValueError(f"Invalid subcontext type: {SUBCONTEXT_TYPE}")

ANSWER_PROMPT_TEMPLATE = """{question}"""
generator_config = LocateFairSectionGenerator.Config(
    answer_client=client,
    answer_temperature=0.0,
    answer_max_completion_tokens=384,
    chunk_size_range=(500, 2000),
    answer_prompt_template=ANSWER_PROMPT_TEMPLATE,
    answer_system_prompt_template=ANSWER_SYSTEM_PROMPT,
    subcontext_generator=subcontext_generator,
)

file_name = Path(__file__).stem


STATE = "Alabama"

config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{STATE.lower()}_n{{num_samples}}_nstatutes{{context.num_statutes}}"),
    convo_generator=generator_config,
    context=HousingStatutesContextConfig(
        states=[STATE],
        split_on="statute",
        num_statutes=100,
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=128,
    max_num_batches_in_parallel=16,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
    ),
)


if __name__ == "__main__":
    pydrantic.main(config)
