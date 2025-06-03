import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.generate.generate_training import GenerateTrainingConfig
from capsules.tasks.finance.reformat_generator import (ReformatGenerator, PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE)
from capsules.tasks.finance import FinanceBenchSectionedContextConfig
from capsules.utils import WandBConfig

file_name = Path(__file__).stem


client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


parallel = True
num_samples = 32678 if parallel else 1
batch_size = min(512, num_samples) if parallel else 1
parallel_batches = min(32, num_samples) if parallel else 1


memorize_config =  GenerateTrainingConfig(

    name=FormatStringVariable(f"{file_name}_n{{num_samples}}"),

    convo_generator=ReformatGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=1024,
        prompt_template=PROMPT_TEMPLATE,
        system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
    ),
    
    context=FinanceBenchSectionedContextConfig(
        doc_name="AMD_2022_10K", 
        max_tokens_per_section=2_000
    ),

    tokenizer="meta-llama/Llama-3.2-3B-Instruct",
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    num_samples=num_samples,
    batch_size=batch_size,
    max_num_batches_in_parallel=parallel_batches,
    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
    ),
)

configs = []
configs.append(memorize_config)


if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])


