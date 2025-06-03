import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient

from capsules.generate.generators.simple_qa import SimpleQuestionFromChunk
from capsules.generate.chunk import SimpleCharacterChunker
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.tasks.swde import SWDEContextConfig
from capsules.tasks.swde.generators import ReformatGenerator, PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE
from capsules.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem

html_page = '0000.htm'
config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{html_page}_n{{num_samples}}"),
    convo_generator=ReformatGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=1024,
        prompt_template=PROMPT_TEMPLATE,
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
    ),
    context=SWDEContextConfig(
            webpage_id=html_page,
            max_tokens_per_section=1_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    # generate config
    num_samples=32768,
    batch_size=256,
    max_num_batches_in_parallel=32,
    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"swde", "generate", f"swde_{html_page}"],
    ),
)

if __name__ == "__main__":
    pydrantic.main([config])

