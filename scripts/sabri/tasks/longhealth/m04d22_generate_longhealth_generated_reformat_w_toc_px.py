import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.tasks.longhealth import LongHealthContextConfig
from cartridges.tasks.longhealth.generators import PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, QUESTION_PROMPT_TEMPLATE, GeneratedReformatWithToCGenerator
from cartridges.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    # url="http://localhost",
    url="https://hazyresearch--tksrs-entry-Cartridges-3b-1xh100-min0-max24-serve.modal.run",
    # ports=[8880 + i for i in range(8)],
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


NUM_PATIENTS = 20
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]
config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{patients_str}_n{{num_samples}}"),
    convo_generator=GeneratedReformatWithToCGenerator.Config(
        question_client=client,
        question_temperature=0.2,
        question_max_completion_tokens=128,
        question_prompt_template=QUESTION_PROMPT_TEMPLATE,
        answer_client=client,
        answer_temperature=0.2,
        answer_max_completion_tokens=1024,
        answer_prompt_template=PROMPT_TEMPLATE,
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        subcontext_type="both",
        subcontext_frac_note=0.8,

        toc_num_samples=64,
        toc_batch_size=4,
        toc_max_num_batches_in_parallel=32,
        toc_max_completion_tokens=128
    ),
    context=LongHealthContextConfig(
        patient_ids=patient_ids,
    ),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    # generate config
    num_samples=32768,
    batch_size=64,
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
