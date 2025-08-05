import os
from pathlib import Path

import pydrantic

from cartridges.clients.openai import OpenAIClient
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
from cartridges.data.mtob.resources import MTOBResource
from cartridges.evaluate import ICLBaseline, EvaluateConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.evaluate import GenerationEvalConfig

from cartridges.utils import WandBConfig

client = OpenAIClient.Config(
    base_url="https://hazyresearch--vllm-qwen3-4b-1xh100-serve.modal.run/v1",
    model_name="Qwen/Qwen3-4b",
)

# client = TokasaurusClient.Config(
#     url="https://hazyresearch--toka-llama-3-1-8b-instruct-1xh100-min0-serve.modal.run",
#     model_name="meta-llama/Llama-3.1-8B-Instruct",
# )
file_name = Path(__file__).stem

SYSTEM_PROMPT_TEMPLATE = f"""You are a helpful and reliable assistant."""

configs = [
    EvaluateConfig(
        name=f"{file_name}",
        generator=ICLBaseline.Config(
            client=client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.0,
            max_completion_tokens=128,
            enable_thinking=False,
            context="",
        ),
        eval=GenerationEvalConfig(
            name_for_wandb=f"mmtob-ke-test",
            dataset=MTOBKalamangToEnglishGenerateDataset.Config(use_cot=False),
            batch_size=16,
            generate_max_new_tokens=128,
            num_samples=1,
            temperature=0.0,
        ),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"mtob", "genbaseline", "icl"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
