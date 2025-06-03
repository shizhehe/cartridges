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
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]



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
    ),
    context=LongHealthStructuredContextConfig(
        patient_ids=patient_ids,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=32 * 64 * 8,# * 32768,
    batch_size=64,
    max_num_batches_in_parallel=64,
    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"mtob", "generate",],
    ),
)


if __name__ == "__main__":
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main([config])