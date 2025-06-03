import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.generate.generators.auto import AutoConvoGenerator, SlicePromptSampler
from capsules.tools.retrieval import BM25RetrieverTool
from capsules.utils import WandBConfig
from capsules.tasks.swde import SWDEContextConfig


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


if "SLICES" in os.environ:
    SLICES = os.environ["SLICES"].split(",")
else:
    SLICES = [
        "structuring",
        "summarization",
        "aggregation",
        "question",
        "use_case",
    ]

html_page = '0000.htm'
config = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_tk_n{{num_samples}}"),
    convo_generator=AutoConvoGenerator.Config(
        client=client,
        tokenizer="meta-llama/Llama-3.2-3B-Instruct",
        max_rounds=1,
        tools=[
        ],
        prompt_sampler=SlicePromptSampler.Config(
            max_tokens_initial_context = 4096,
            slices = SLICES,
        ),
    ),
    context=SWDEContextConfig(
            webpage_id=html_page,
            max_tokens_per_section=1_000,
    ),

    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),

    num_samples=num_samples,
    batch_size=min(num_samples, 256) if parallel else 1,
    max_num_batches_in_parallel=min(num_samples, 16) if parallel else 1,

    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"swde", "auto_generate"],
    ),
)


if __name__ == "__main__":
    pydrantic.main([config])

