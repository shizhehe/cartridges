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
    url="https://hazyresearch--toka-llama-3-2-3b-1xh100-batch-serve.modal.run",
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
# VT_PATH = f"{BASE_PATH}/qwen3_4b-l100000-n1-c64-h2-noise-9df65ada.json"
# VT_PATH = f"{BASE_PATH}/qwen3_4b-l100000-n1-c64-h2-essay-979310a3.json"
# VT_PATH = f"{BASE_PATH}/llama_3.2_3b_instruct-l100000-n1-c64-h2-essay-7ba69bcb.json"

# Natural language
# VT_PATH = f"{BASE_PATH}/llama_3.2_3b_instruct-l10000-n1-c16-h2-essay-words-1d31e1f5.json"
VT_PATH = f"{BASE_PATH}/llama_3.2_3b_instruct-l100000-n1-c64-h2-essay-words-3e9c7e72.json"

THINKING = False

configs = [
    EvaluateConfig(
        name=f"{file_name}",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.0,
            max_completion_tokens=1024,
            context=VariableTrackingResource.Config(variable_tracking_path=VT_PATH),
            enable_thinking=THINKING,
        ),
        eval=GenerationEvalConfig(
            dataset=VariableTrackingGenerateDataset.Config(
                variable_tracking_path=VT_PATH,
                thinking=THINKING,
            ),
            name_for_wandb=f"variable_tracking",
            num_samples=1,
            temperature=0.0,
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