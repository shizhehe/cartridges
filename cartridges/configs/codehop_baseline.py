import os
from pathlib import Path

import pydrantic

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.clients.openai import OpenAIClient
from cartridges.data.codehop.evals import CodeHopGenerateDataset
from cartridges.data.resources import DirectoryResource
from cartridges.evaluate import ICLBaseline, EvaluateConfig
from cartridges.evaluate import GenerationEvalConfig

from cartridges.utils.wandb import WandBConfig

from cartridges.configs.codehop_synthesize import DATASET_DIR
dataset_dir = Path(DATASET_DIR).parent

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-llama-3-2-3b-1xh100-batch-serve.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

# client = TokasaurusClient.Config(
#     url="https://hazyresearch--toka-llama-3-1-8b-1xh100-batch-serve.modal.run",
#     model_name="meta-llama/Llama-3.1-8B-Instruct",
# )

# client = TokasaurusClient.Config(
#     url="https://hazyresearch--toka-qwen3-4b-1xh100-min0-serve.modal.run",
#     model_name="Qwen/Qwen3-4b",
# )

# client = OpenAIClient.Config(
#     model_name="gpt-4o-mini",
# )

SYSTEM_PROMPT_TEMPLATE = """Below is a list of Python files.
Each file contains a number of methods. Note, that some of the methods may make calls to other
methods which are defined in other files.

{content}
"""


configs = [
    EvaluateConfig(
        name=f"codehop_baseline",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.0,
            max_completion_tokens=128,
            enable_thinking=False,
            context=DirectoryResource.Config(
                path=DATASET_DIR,
                seed_prompts=[]
            )
        ),
        eval=GenerationEvalConfig(
            dataset=CodeHopGenerateDataset.Config(make_run_dir=str(dataset_dir)),
            name_for_wandb=f"codehop",
            generate_max_new_tokens=8,
            temperature=0.0,
        ),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(tags=[f"codehop", "genbaseline", "icl"]),
        output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
