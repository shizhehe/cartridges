import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.generators.auto import AutoConvoGenerator, SlicePromptSamplerWithChunks
from cartridges.tasks.longhealth import LongHealthContextConfig
from cartridges.tasks.mtob.context import MTOBNoStructuredContext
from cartridges.utils import WandBConfig

from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-8b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

file_name = Path(__file__).stem

SLICES = [
    "generic",
]

config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_s{len(SLICES)}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        max_rounds=1,
        prompt_sampler=SlicePromptSamplerWithChunks.Config(
            slices=SLICES,
            min_chunk_size=512,
            max_chunk_size=4096,
            desc="The following is an excerpt from a grammar book about the Kalamang language."
        ),
        prob_cot_a=0.2,
        use_tools_a=False,
        use_tools_b=False,
        tools=[]
    ),
    context=MTOBNoStructuredContext(setup="medium_and_sentences"),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536,
    batch_size=128,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"mtob", "generate", "paper"],
    ),
)


if __name__ == "__main__": 
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
