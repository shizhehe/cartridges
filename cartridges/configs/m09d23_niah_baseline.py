import os
from pathlib import Path

import pydrantic

from cartridges.clients.openai import OpenAIClient
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.data.ruler.evals import NIAHGenerateDataset
from cartridges.data.ruler.resources import NIAHResource
from cartridges.evaluate import ICLBaseline, GenerationEvalRunConfig, ICLBaselineSummaryFromLargeModel
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.evaluate import GenerationEvalConfig

from cartridges.utils.wandb import WandBConfig

# client = OpenAIClient.Config(
#     base_url="https://hazyresearch--vllm-qwen3-4b-1xh100-serve.modal.run/v1",
#     model_name="Qwen/Qwen3-4b",
# )


client = TokasaurusClient.Config(
    url=os.environ["CARTRIDGES_TOKASAURUS_LLAMA_3_2_3B_URL"],
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem

SYSTEM_PROMPT_TEMPLATE = f"""Please answer the question below about the following context.

<context>
{{content}}
</context>
"""

BASE_PATH = os.path.join(
    os.environ["CARTRIDGES_DIR"], "cartridges/data/ruler/_data"
)

NUM_KEYS = (1, 1)
num_keys_str = f"k{NUM_KEYS[0]}_{NUM_KEYS[1]}"

# enable_thinking = not (NUM_KEYS == (1, 1))

NUM_KEYS_TO_PATH = {
    (1, 1): f"{BASE_PATH}/llama_3.2_3b_instruct-l128000-n1-k128-v1_1-essay-key_words-val_numbers-74198fb4.json"
}

NIAH_PATH = NUM_KEYS_TO_PATH[NUM_KEYS]

TEMPERATURE = 0.3

baselines = []

icl_baseline = ICLBaseline.Config(
    client=client,
    system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
    temperature=TEMPERATURE,
    max_completion_tokens=256,
    context=NIAHResource.Config(niah_path=NIAH_PATH),
)
# baselines.append(icl_baseline)

# --- begin summary baselines ----
for summary_tokens in [128, 256, 512, 1024, 2048]:
    summary_baseline = ICLBaselineSummaryFromLargeModel.Config(
        client=client,
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        summary_client=OpenAIClient.Config(
            model_name="gpt-5",
        ),
        summary_tokens=summary_tokens,
        temperature=TEMPERATURE,
        max_completion_tokens=256,
        context=NIAHResource.Config(niah_path=NIAH_PATH),
    )
    # baselines.append((f"summary_{summary_tokens}", summary_baseline))
# --- end summary baselines ----

# --- begin truncation baselines ----
for max_context_tokens in [65_536, 32_768, 16_384, 8_192, 4_096, 2_048]:
    icl_baseline = ICLBaseline.Config(
        client=client,
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
        temperature=TEMPERATURE,
        max_completion_tokens=256,
        max_context_tokens=max_context_tokens,
        context=NIAHResource.Config(niah_path=NIAH_PATH),
    )
    baselines.append(
        (f"truncate_{max_context_tokens}", icl_baseline)
    )

# --- end truncation baselines ----


configs = [
    GenerationEvalRunConfig(
        name=f"{file_name}_{num_keys_str}_{suffix}",
        generator=baseline,
        eval=GenerationEvalConfig(
            dataset=NIAHGenerateDataset.Config(
                niah_path=NIAH_PATH,
                thinking=False,
            ),
            name_for_wandb=f"niah_mc",
            num_samples=1,
            temperature=TEMPERATURE,
        ),
        max_num_batches_in_parallel=4,
        batch_size=32,
        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "niah", num_keys_str],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    )
    for suffix, baseline in baselines
]

if __name__ == "__main__":
    pydrantic.main(configs)