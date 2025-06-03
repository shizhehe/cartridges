import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable


from capsules.clients.openai_batch import TokasaurusClient
from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.generators.memorization import LocateFairSectionGenerator
from capsules.generate.subcontext import RandomSubcontextGenerator, RandomizedSlidingWindowGenerator
from capsules.tasks.longhealth import LongHealthContextConfig
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.utils import WandBConfig


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(8)],
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


ANSWER_SYSTEM_PROMPT = f"""You are a medical expert. 
Please use the following information from a medical record to answer the question.

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
        max_num_chunks=3,
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


PATIENT_IDXS = [1, 2, 3]
patients_str = ''.join(f"p{idx:02d}" for idx in PATIENT_IDXS)  # used for names and tags
patient_ids = [f"patient_{idx:02d}" for idx in PATIENT_IDXS]
config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{patients_str}_n{{num_samples}}"),
    convo_generator=generator_config,
    context=LongHealthContextConfig(
        patient_ids=patient_ids,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32768,
    batch_size=128,
    max_num_batches_in_parallel=16,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"longhealth", "generate", f"patients_{patients_str}"],
    ),
)


if __name__ == "__main__":
    pydrantic.main(config)
