import os
from pathlib import Path
import uuid


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
from cartridges.tools.retrieval import BM25RetrieverTool, OpenAIRetrieverTool
from cartridges.utils import WandBConfig

from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-8b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    max_retries=25,
)

file_name = Path(__file__).stem

SLICES = [
    "structuring",
    "summarization",
    "question",
    "use_case",
    "creative",
]

NUM_PROCESSES_PER_SLICE = 1

# important to identify groups of runs that are the same 
unique_id = str(uuid.uuid4())[:6]

configs = [
    GenerateTrainingConfig(
        name=FormatStringVariable(f"{file_name}_{unique_id}_{slice}_n{{num_samples}}"),
        run_id=FormatStringVariable("{name}"),
        convo_generator=AutoConvoGenerator.Config(
            client=client,
            tokenizer="meta-llama/Llama-3.1-8B-Instruct",
            max_rounds=1,
            prompt_sampler=SlicePromptSamplerWithChunks.Config(
                slices=[slice],
                min_chunk_size=512,
                max_chunk_size=4096,
                desc="The following is an excerpt from a grammar book about the Kalamang language.",
            ),
            prob_cot_a=0.2,
            use_tools_b=True,
            use_tools_a=False,
            tools=[
                OpenAIRetrieverTool.Config(
                    max_tokens_per_chunk=128,
                    embedding_model="text-embedding-3-large",
                    cache_dir=os.path.join(
                        os.environ["CARTRIDGES_OUTPUT_DIR"], "openai-embeddings-cache"
                    ),
                    max_top_k=10,
                ),
            ],
        ),
        context=MTOBNoStructuredContext(setup="latex_and_sentences"),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        num_samples=65536 // (NUM_PROCESSES_PER_SLICE * len(SLICES)),
        batch_size=128,
        max_num_batches_in_parallel=128 // (NUM_PROCESSES_PER_SLICE * len(SLICES)),
        parallelism_strategy="thread",
        save_wandb_artifact=True,
        wandb=WandBConfig(
            project="cartridges",
            entity="hazy-research",
            tags=[f"mtob", "generate", "paper"],
        ),
    )
    for slice in SLICES
    for _ in range(NUM_PROCESSES_PER_SLICE)
]


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main(configs)
