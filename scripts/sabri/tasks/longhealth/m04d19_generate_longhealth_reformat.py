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
from cartridges.tasks.longhealth.generators import ReformatGenerator, PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE
from cartridges.tasks.reglab.housing_qa import HousingStatutesContextConfig
from cartridges.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    # url="http://localhost",
    url="https://hazyresearch--tksrs-entry-Cartridges-llama-3b-1xh100-min0-serve.modal.run",
    # ports=[8880 + i for i in range(8)],
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


PATIENT_IDXS = [1, 2, 3]
patients_str = ''.join(f"p{idx:02d}" for idx in PATIENT_IDXS)  # used for names and tags
patient_ids = [f"patient_{idx:02d}" for idx in PATIENT_IDXS]
config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{patients_str}_n{{num_samples}}"),
    convo_generator=ReformatGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=1024,
        prompt_template=PROMPT_TEMPLATE,
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        subcontext="patient",
    ),
    context=LongHealthContextConfig(
        patient_ids=patient_ids,
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    # generate config
    num_samples=32768,
    batch_size=256,
    max_num_batches_in_parallel=32,
    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"longhealth", "generate", f"patients_{patients_str}"],
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
