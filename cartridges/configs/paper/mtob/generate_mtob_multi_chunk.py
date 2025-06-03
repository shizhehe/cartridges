import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.clients.tokasaurus_batch import TokasaurusBatchClient

from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.generators.auto import (
    AutoConvoGenerator,
    SlicePromptSampler,
    SlicePromptSamplerWithChunks,
    SlicePromptSamplerWithSummarization,
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

from capsules.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-8b-1xh100-min0-max32-serve.modal.run",
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
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=65536,
    batch_size=128,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"mtob", "generate", "paper"],
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])
