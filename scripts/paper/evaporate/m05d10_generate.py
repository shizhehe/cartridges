import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.generators.auto import AutoConvoGenerator, SlicePromptSamplerWithChunks
from cartridges.tools.retrieval import BM25RetrieverTool
from cartridges.utils import WandBConfig
from cartridges.tasks.fda import EvaporateContextConfig, EvaporateMultipleChoiceGenerateDataset, EvaporateEvalDataset


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)
file_name = Path(__file__).stem


configs = []
bs = 64
num_samples = 65536
parallel = True


SLICES = [
   "structuring",
#    "summarization",
   "question",
#    "use_case",
#    "creative",
]

for DOC_NAME in [
    "K152386.txt"
    #  "K182513.txt", "K181324.txt", 
    # "K173887.txt"
]:

    config = GenerateTrainingConfig(
        name=FormatStringVariable(f"{file_name}_doc{DOC_NAME}_n{{num_samples}}"),
        convo_generator=AutoConvoGenerator.Config(
            client=client,
            tokenizer="meta-llama/Llama-3.2-3B-Instruct",
            max_rounds=1,
            tools=[],
            use_tools = False,
            prob_cot_a = 0.2,
            prompt_sampler=SlicePromptSamplerWithChunks.Config(
                min_chunk_size = 512,
                max_chunk_size = 4096,

                slices = SLICES,
                desc = "This document is an FDA report for a single medical device. Information needs to be extracted from this document in order to populate a database of different devices.",
            ),
        ),
        context=EvaporateContextConfig(
            doc_id=DOC_NAME,
            max_tokens_per_section=2048,
        ),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),

        num_samples=num_samples,
        batch_size=min(num_samples, 256) if parallel else 1,
        max_num_batches_in_parallel=min(num_samples, 16) if parallel else 1,

        save_wandb_artifact=False,
        wandb=WandBConfig(
            project="cartridges",
            entity="hazy-research",
            tags=[f"fda", "generate", f"doc{DOC_NAME}"],
        ),
    )
    configs.append(config)


if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])

