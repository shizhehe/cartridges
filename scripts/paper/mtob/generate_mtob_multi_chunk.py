import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.generators.auto import (
    AutoConvoGenerator,
    SlicePromptSampler,
    SlicePromptSamplerWithChunks,
    SlicePromptSamplerWithSummarization,
    TreeSlicePromptSampler,
)
from cartridges.tasks.longhealth import LongHealthContextConfig
from cartridges.tasks.longhealth.generators import (
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    QUESTION_PROMPT_TEMPLATE,
    GeneratedReformatWithToCGenerator,
)
from cartridges.tasks.longhealth.seed_prompts import (
    LongHealthPromptSampler,
    LongHealthTreePromptSampler,
)
from cartridges.tasks.mtob.context import MTOBNoStructuredContext
from cartridges.tools.retrieval import BM25RetrieverTool
from cartridges.utils import WandBConfig

from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-Cartridges-8b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
)

file_name = Path(__file__).stem

SLICES = [
    "structuring",
    "summarization",
    "question",
    "use_case",
    "creative",
]

config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_s{len(SLICES)}_n{{num_samples}}"),
    run_id=FormatStringVariable("{name}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        max_rounds=1,
        prompt_sampler=TreeSlicePromptSampler.Config(
            slices=SLICES,
            max_tokens_per_page=300,
            max_tokens_in_context=(9_000, 10_000),
            num_focus_leaves_per_context=6,
            sibling_bias=3,
            desc="The following is an excerpt from a grammar book about the Kalamang language.",
        ),
        prob_cot_a=0.2,
        use_tools=False,
        tools=[],
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
