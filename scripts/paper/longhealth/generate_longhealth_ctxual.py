import os
from pathlib import Path


import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient

from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.generators.auto import AutoConvoGenerator, SlicePromptSamplerWithContextualizedChunks
from cartridges.tools.retrieval import OpenAIRetrieverTool
from cartridges.utils import WandBConfig

from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-Cartridges-3b-1xh100-min0-max64-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    max_retries=1
)

file_name = Path(__file__).stem


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

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

NUM_PROCESSES = 1

name = f"{file_name}_{patients_str}_s{len(SLICES)}_n{{num_samples}}"

configs = [
    GenerateTrainingConfig(
        name=FormatStringVariable(name),
        run_id=FormatStringVariable(name),
        convo_generator=AutoConvoGenerator.Config(
            client=client,
            tokenizer="meta-llama/Llama-3.2-3B-Instruct",
            max_rounds=1,
            prompt_sampler=SlicePromptSamplerWithContextualizedChunks.Config(
                slices=SLICES,
                min_chunk_size=512,
                max_chunk_size=2048,
                num_summaries=32,
                summary_temperature=0.3,
            ),
            prob_cot_a=0.2,
            use_tools_b=False, 
            use_tools_a=False,
            tools=[
                OpenAIRetrieverTool.Config(
                    max_tokens_per_chunk=1024,
                    embedding_model="text-embedding-3-large",
                    cache_dir=os.path.join(os.environ["CARTRIDGES_OUTPUT_DIR"], "openai-embeddings-cache"),
                    max_top_k=3
                ),
            ]
        ),
        context=LongHealthStructuredContextConfig(patient_ids=patient_ids),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        num_samples=65536 // NUM_PROCESSES,
        batch_size=128,
        max_num_batches_in_parallel=128 // NUM_PROCESSES,
        parallelism_strategy="thread",
        save_wandb_artifact=True,
        wandb=WandBConfig(
            project="cartridges",
            entity="hazy-research",
            tags=[f"longhealth", "generate", f"patients_{patients_str}", "paper"],
        ),
    )
    for _ in range(NUM_PROCESSES)
]


if __name__ == "__main__": 
    # doc_name = DOC_NAMES[int(os.environ["DOC_INDEX"])]
    pydrantic.main(configs)
