import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient

from capsules.context import BaseContextConfig, TexDocument
from capsules.data.hydragen import HydragenStructuredContext
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.generators.auto import (
    SLICE_TYPES,
    AutoConvoGenerator,
    SlicePromptSampler,
    TreeSlicePromptSampler,
)
from capsules.tasks.longhealth import LongHealthContextConfig
from capsules.tasks.longhealth.generators import (
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    QUESTION_PROMPT_TEMPLATE,
    GeneratedReformatWithToCGenerator,
)
from capsules.tasks.longhealth.seed_prompts import (
    LongHealthPromptSampler,
    LongHealthTreePromptSampler,
)
from capsules.tasks.mtob.context import MTOBNoStructuredContext
from capsules.tools.retrieval import BM25RetrieverTool
from capsules.utils import WandBConfig


client = TokasaurusBatchClient.Config(
    # url="http://localhost",
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    # ports=[8880 + i for i in range(8)],
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    timeout=2000,
)

file_name = Path(__file__).stem

SLICES: list[SLICE_TYPES] = [
    "summarization",
    "question",
    "use_case",
    "creative",
]



config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{"_".join(SLICES)}_{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        # prompt_sampler=SlicePromptSampler.Config(
        #     max_tokens_initial_context=4096,
        #     slices=SLICES,
        # ),
        prompt_sampler=TreeSlicePromptSampler.Config(
            slices=SLICES,
            max_tokens_per_page=300,
            max_tokens_in_context=5_000,
            num_focus_leaves_per_context=6,
            sibling_bias=3,
        ),
        use_tools=False,
        tools=[],
        temperature_a=0.8
    ),
    context=HydragenStructuredContext(),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=128 * 32 * 3,
    batch_size=128,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[
            f"mtob",
            "generate",
        ],
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
