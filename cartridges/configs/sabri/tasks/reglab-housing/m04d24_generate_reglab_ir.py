import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient

from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.tasks.reglab.housing_qa import HousingStatutesContextConfig
from capsules.tasks.reglab.generators import SYSTEM_PROMPT_TEMPLATE, QUESTION_PROMPT_TEMPLATE, ANSWER_PROMPT_TEMPLATE, LegalInstructionResponseGenerator
from capsules.tasks.reglab.utils import ALL_STATES
from capsules.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    # url="http://localhost",
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    # ports=[8880 + i for i in range(8)],
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


NUM_STATES = 5
states_str = f"{NUM_STATES}states"
states = ALL_STATES[:NUM_STATES]
config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{states_str}_n{{num_samples}}"),
    convo_generator=LegalInstructionResponseGenerator.Config(
        question_client=client,
        question_temperature=0.2,
        question_max_completion_tokens=128,
        question_prompt_template=QUESTION_PROMPT_TEMPLATE,

        answer_client=client,
        answer_temperature=0.2,
        answer_max_completion_tokens=1024,
        answer_prompt_template=ANSWER_PROMPT_TEMPLATE,

        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        subcontext_type="intra_state",
        subcontext_frac_inter=0.8,
    ),
    context=HousingStatutesContextConfig(
        states=states,
        split_on="statute",
        num_extra_statutes=0
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=32768,
    batch_size=64,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"reglab-housing", "generate", f"{states_str}"],
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
