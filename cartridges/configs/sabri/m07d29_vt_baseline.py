import os
from pathlib import Path

import pydrantic

from cartridges.clients.openai import OpenAIClient
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.ruler.evals import VariableTrackingGenerateDataset
from cartridges.data.ruler.resources import VariableTrackingResource
from cartridges.evaluate import ICLBaseline, EvaluateConfig
from cartridges.evaluate import GenerationEvalConfig

from cartridges.utils import WandBConfig

client = TokasaurusClient.Config(
    url="https://hazyresearch--toka-llama-3-2-3b-instruct-1xh100-min0-serve.modal.run",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem

SYSTEM_PROMPT_TEMPLATE = f"""Please answer the question below about the following context.

<context>
{{content}}
</context>
"""

BASE_PATH = "/home/sabri/code/cartridges/cartridges/data/ruler/_data"

# Use the default variable tracking dataset path
VT_PATH = f"{BASE_PATH}/qwen3_4b-l100000-n1-c128-h3-noise-6fe0da10.json"

configs = [
    EvaluateConfig(
        name=f"{file_name}",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.3,
            max_completion_tokens=256,
            context=VariableTrackingResource.Config(variable_tracking_path=VT_PATH),
        ),
        eval=GenerationEvalConfig(
            dataset=VariableTrackingGenerateDataset.Config(
                variable_tracking_path=VT_PATH,
                thinking=True,
            ),
            name_for_wandb=f"variable_tracking",
            num_samples=1,
            temperature=0.3,
        ),
        max_num_batches_in_parallel=4,
        batch_size=32,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"variable_tracking", "genbaseline", "icl"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)