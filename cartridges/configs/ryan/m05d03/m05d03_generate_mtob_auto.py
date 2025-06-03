import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient

from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.generators.auto_with_walks_and_focus_text import AutoConvoGenerator
from capsules.tasks.longhealth import LongHealthContextConfig
from capsules.tasks.longhealth.generators import PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE, QUESTION_PROMPT_TEMPLATE, GeneratedReformatWithToCGenerator
from capsules.tasks.mtob.context import MTOBStructuredContextConfig
from capsules.tools.retrieval import BM25RetrieverTool
from capsules.utils import WandBConfig

from capsules.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    url="http://localhost",
    ports=[8880 + i for i in range(4)],
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    timeout=1500,
)

file_name = Path(__file__).stem


NUM_PATIENTS = 10

config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.1-8B-Instruct",
        max_rounds=1,
        tools=[
            BM25RetrieverTool.Config(
                max_tokens_per_chunk=256,
            ),
            BM25RetrieverTool.Config(
                max_tokens_per_chunk=1024,
            ),
            # OpenAIRetrieverTool.Config(
            #     max_tokens_per_chunk=256,
            #     embedding_model="text-embedding-3-large",
            # ),
        ],
        num_focus_leaves_per_context=32,
        seed=59,
    ),
    context=MTOBStructuredContextConfig(
        # book_type="full",
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=16384,
    batch_size=64*4,
    max_num_batches_in_parallel=1,
    save_wandb_artifact=True,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"mtob", "generate",],
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])