import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

from cartridges.generate.generators.simple_qa import SimpleQuestionFromChunk
from cartridges.generate.chunk import SimpleCharacterChunker
from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.subcontext import RandomSubcontextGenerator, RandomizedSlidingWindowGenerator
from cartridges.tasks.longhealth import LongHealthContextConfig
from cartridges.tasks.reglab.housing_qa import HousingStatutesContextConfig
from cartridges.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    # url="http://localhost",
    url="https://hazyresearch--tksrs-entry-capsules-llama-3b-1xh100-min1-serve.modal.run",
    # ports=[8880 + i for i in range(8)],
    ports=None,
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

PATIENT_IDXS = [1, 2, 3]
patients_str = ''.join(f"p{idx:02d}" for idx in PATIENT_IDXS)  # used for names and tags
patient_ids = [f"patient_{idx:02d}" for idx in PATIENT_IDXS]
config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{patients_str}_n{{num_samples}}"),
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
    context=LongHealthContextConfig(
        patient_ids=patient_ids,
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    # generate config
    num_samples=32768,
    batch_size=256,
    max_num_batches_in_parallel=32,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"longhealth", "generate", f"patients_{patients_str}"],
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
