import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient

from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.run import GenerateConfig
from capsules.generate.subcontext import RandomSubcontextGenerator, RandomizedSlidingWindowGenerator
from capsules.tasks.longhealth import LongHealthContextConfig
from capsules.tasks.reglab.housing_qa import (
    HousingStatutesContextConfig,
    StatuteRecallEvalGenerator,
)
from capsules.utils import WandBConfig

client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

STATE = "Alabama"


QUESTION_PROMPT_TEMPLATE = f"""Please generate a challenging question about the following excerpt from a hospital's medical records. 

---

{{chunk}}

---
The question should test a model's knowledge of the information in the document when asked in a closed-book setting.
Output only a single question. Do NOT include any other text or explanation other than the question."""

QUESTION_SYSTEM_PROMPT = f"""You are working to train a language model on a hospital's medical records.

You will be given excerpts of a hospital's medical records and your job is to generate a training question that is grounded in the provided medical records. 
After, we will train a langauge model to answer the question without access to the medical records (i.e. closed book). 
The question should be challenging and can require the model to remember specific medical details. 
"""


ANSWER_SYSTEM_PROMPT = f"""You are a medical expert. 
Please use the following information from a medical record to answer the question.

<excerpt>
{{subcontext}}
</excerpt>

First, think step-by-step. Then provide your answer to the question. 
Do NOT begin your answer with "According to the excerpt..." or any other similar phrase.
"""

ANSWER_PROMPT_TEMPLATE = """{question}"""

file_name = Path(__file__).stem


subcontext_generator = RandomSubcontextGenerator.Config(
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    min_num_chunks=1,
    max_num_chunks=6,
    num_contexts=16384,
    seed=32,
)


config = GenerateConfig(
    name=FormatStringVariable(f"{file_name}_{STATE}_n{{num_samples}}"),
    convo_generator=StatuteRecallEvalGenerator.Config(
        states=[STATE],
    ),
    context=HousingStatutesContextConfig(states=[STATE]),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=45,
    batch_size=256,
    max_num_batches_in_parallel=64,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"reglab", "generate"],
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
