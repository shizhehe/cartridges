import os
from pathlib import Path

import pydrantic

from cartridges.clients.openai import OpenAIClient
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.data.ruler.evals import NIAHGenerateDataset
from cartridges.data.ruler.resources import NIAHResource
from cartridges.evaluate import ICLBaseline, EvaluateConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.evaluate import GenerationEvalConfig

from cartridges.utils import WandBConfig

# client = OpenAIClient.Config(
#     base_url="https://hazyresearch--vllm-qwen3-4b-1xh100-serve.modal.run/v1",
#     model_name="Qwen/Qwen3-4b",
# )


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
NIAH_PATH = f"{BASE_PATH}/qwen3_4b-l100000-n1-k128-v1_2-essay-key_words-val_numbers--1660737731696865120.json"
NIAH_PATH = f"{BASE_PATH}/qwen3_4b-l100000-n1-k128-v1_1-essay-key_words-val_numbers-e83970e8.json"


configs = [
    EvaluateConfig(
        name=f"{file_name}",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.0,
            max_completion_tokens=128,
            context=NIAHResource.Config(niah_path=NIAH_PATH),
        ),
        eval=GenerationEvalConfig(
            dataset=NIAHGenerateDataset.Config(
                niah_path=NIAH_PATH,
                thinking=False,
            ),
            name_for_wandb=f"niah_mc",
            num_samples=1,
            temperature=0.0,
        ),
        max_num_batches_in_parallel=4,
        batch_size=32,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"longhealth", "genbaseline", "icl"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
