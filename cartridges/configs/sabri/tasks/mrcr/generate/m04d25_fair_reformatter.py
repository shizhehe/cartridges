import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.tasks.swde.decomp_generate_training import GenerateTrainingConfig
from capsules.tasks.mrcr import MRCRContextConfig
from capsules.tasks.mrcr.fairer_generator import DirectGenerator, ComparePositionsGenerator, PrecedingChatGenerator, NextChatGenerator, CheatGenerator
from capsules.utils import WandBConfig

client = TokasaurusBatchClient.Config(
    url="https://hazyresearch--tksrs-entry-capsules-3b-1xh100-min0-max32-serve.modal.run",
    ports=None,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem


doc = '0'
configs = []
parallel = True
expt_tag = ""


# directly generate training data
config_1 = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_direct{expt_tag}"),
    convo_generator=DirectGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=10,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 1,
    batch_size=256 if parallel else 1,
    max_num_batches_in_parallel=32 if parallel else 1,

    # stage 1 categories
    num_samples_stage_1=1,
    batch_size_stage_1=1,
    max_num_batches_in_parallel_stage_1=1,

    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"mrcr", "generate", f"mrcr_{doc}"],
    ),
)
# configs.append(config_1)


# generate comparison with other sections
config_2 = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_compare{expt_tag}"),
    convo_generator=ComparePositionsGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=10,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 1,
    batch_size=256 if parallel else 1,
    max_num_batches_in_parallel=32 if parallel else 1,

    # stage 1 categories
    num_samples_stage_1=1,
    batch_size_stage_1=1,
    max_num_batches_in_parallel_stage_1=1,

    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"mrcr", "generate", f"mrcr_{doc}"],
    ),
)
# configs.append(config_2)


# preceding chat generator
config_3 = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_preceding{expt_tag}"),
    convo_generator=PrecedingChatGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=10,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 1,
    batch_size=256 if parallel else 1,
    max_num_batches_in_parallel=32  if parallel else 1,

    # stage 1 categories
    num_samples_stage_1=1,
    batch_size_stage_1=1,
    max_num_batches_in_parallel_stage_1=1,

    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"mrcr", "generate", f"mrcr_{doc}"],
    ),
)
# configs.append(config_3)


# next chat generator
config_4 = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_next{expt_tag}"),
    convo_generator=NextChatGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=10,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 1,
    batch_size=256 if parallel else 1,
    max_num_batches_in_parallel=32  if parallel else 1,

    # stage 1 categories
    num_samples_stage_1=1,
    batch_size_stage_1=1,
    max_num_batches_in_parallel_stage_1=1,

    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"mrcr", "generate", f"mrcr_{doc}"],
    ),
)
# configs.append(config_4)


# next chat generator
config_5 = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_cheat{expt_tag}"),
    convo_generator=CheatGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=10,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 1,
    batch_size=256 if parallel else 1,
    max_num_batches_in_parallel=1  if parallel else 1,

    # stage 1 categories
    num_samples_stage_1=1,
    batch_size_stage_1=1,
    max_num_batches_in_parallel_stage_1=1,

    save_wandb_artifact=False,
    wandb=WandBConfig(
        project="capsules",
        entity="hazy-research",
        tags=[f"mrcr", "generate", f"mrcr_{doc}"],
    ),
)
configs.append(config_5)


if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])

