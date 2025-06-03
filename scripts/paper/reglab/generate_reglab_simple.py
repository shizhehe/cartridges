import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.generators.auto import AutoConvoGenerator, SlicePromptSamplerWithChunks
from cartridges.tasks.reglab.context import ReglabHousingStructuredContextConfig
from cartridges.tasks.reglab.utils import ALL_STATES
from cartridges.utils import WandBConfig



client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


NUM_STATES = 5
states_str = f"{NUM_STATES}states"
states = ALL_STATES[:NUM_STATES]


if "SLICES" in os.environ:
    SLICES = os.environ["SLICES"].split(",")
else:
    SLICES = [
        "structuring",
        "summarization",
        "question",
        "use_case",
        "creative",
    ]

config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{states_str}_s{len(SLICES)}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        prompt_sampler=SlicePromptSamplerWithChunks.Config(
            slices=SLICES,
            min_chunk_size=512,
            max_chunk_size=4096,
            desc=f"Below is a section of a state's legal code. It is part of a larger corpus of legal codes for different states."
        ),
        prob_cot_a=0.2,
        use_tools=False, 
        tools=[]
    ),
    context=ReglabHousingStructuredContextConfig(
        states=states,
    ),    
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    num_samples=65536,
    batch_size=128,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="cartridges",
        entity="hazy-research",
        tags=[f"reglab-housing", "generate", f"{states_str}"],
    ),
)


if __name__ == "__main__": 
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
