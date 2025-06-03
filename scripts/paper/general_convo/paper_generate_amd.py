import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.clients.tokasaurus_batch import TokasaurusBatchClient
from cartridges.generate.generate_training import GenerateTrainingConfig
from cartridges.generate.generators.auto import AutoConvoGenerator, SlicePromptSampler
from cartridges.tools.retrieval import BM25RetrieverTool
from cartridges.utils import WandBConfig
from cartridges.tasks.finance import FinanceBenchContextConfig, FinanceBenchDocumentStructuredConfig


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-Cartridges-3b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


configs = []
DOC_NAME = "AMD_2022_10K"
bs = 64
num_samples = 8192
parallel = True

configs = []
for slice in [
    "structuring",
    "summarization",
    "aggregation",
    "question",
    "use_case",
]:

    config = GenerateTrainingConfig(
        name=FormatStringVariable(f"{file_name}_doc{DOC_NAME}_n{{num_samples}}_slice{slice}"),
        convo_generator=AutoConvoGenerator.Config(
            client=client,
            tokenizer="meta-llama/Llama-3.2-3B-Instruct",
            max_rounds=1,
            use_tools=False,
            prob_cot_a=0.3,
            tools=[
                BM25RetrieverTool.Config(
                    max_tokens_per_chunk=256,
                ),
                BM25RetrieverTool.Config(
                    max_tokens_per_chunk=1024,
                ),
            ],
            prompt_sampler=SlicePromptSampler.Config(
                max_tokens_initial_context=8192,
                slices=[slice],
            ),
        ),
        context=FinanceBenchDocumentStructuredConfig(
            doc_names=[DOC_NAME], 
        ),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
        num_samples=num_samples,
        batch_size=min(num_samples, 256) if parallel else 1,
        max_num_batches_in_parallel=min(num_samples, 32) if parallel else 1,
        save_wandb_artifact=True,
        wandb=WandBConfig(
            project="cartridges",
            entity="hazy-research",
            tags=[f"finance", "generate", f"doc{DOC_NAME}"],
        ),
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
