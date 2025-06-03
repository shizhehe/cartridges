import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.clients.tokasaurus_batch import TokasaurusBatchClient
from capsules.tasks.swde.decomp_generate_training import GenerateTrainingConfig
from capsules.tasks.mrcr import MRCRContextConfig
from capsules.tasks.mrcr.fairer_generator import ComparePositionsGenerator, PrecedingChatGenerator, NextChatGenerator, CheatGenerator, ModifyGenerator, SimpleQAHistoryGenerator, HistoryQAGenerator, RecallPassageGenerator
from capsules.tasks.mrcr.fairer_generator_attempt_v2 import CategoryGenerator, SimpleQAGenerator, DirectGenerator
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
config_direct = GenerateTrainingConfig(
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
    num_samples=32768 if parallel else 16,
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

# generate comparison with other sections
config_compare = GenerateTrainingConfig(
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
    num_samples=32768 if parallel else 16,
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

# preceding chat generator
config_preceding = GenerateTrainingConfig(
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
    num_samples=32768 if parallel else 16,
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

# next chat generator
config_next = GenerateTrainingConfig(
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
    num_samples=32768 if parallel else 16,
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

# next chat generator
config_cheat = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_cheat{expt_tag}_fullHistory_prefixRand"),
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
    num_samples=32768 if parallel else 16,
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

# simple qa generator
config_simpleqa = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_simpleqa{expt_tag}_variant"),
    convo_generator=SimpleQAGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=512,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 16,
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

# modify generator
config_modify = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_modify{expt_tag}"),
    convo_generator=ModifyGenerator.Config(
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
    num_samples=100,
    batch_size=1,
    max_num_batches_in_parallel=32,

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

# simple qa history
config_simpleqa_history = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_simpleqa_history{expt_tag}"),
    convo_generator=SimpleQAHistoryGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=512,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 16,
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

# history qa
config_historyqa = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_historyqa{expt_tag}"),
    convo_generator=HistoryQAGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=512,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 16,
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

# recall history
config_recallqa = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_recallqa{expt_tag}"),
    convo_generator=RecallPassageGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=512,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=32768 if parallel else 16,
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


# attempt
config_attempt = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_{doc}_n{{num_samples}}_attempt{expt_tag}"),
    convo_generator=CategoryGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=1024,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=doc,
        max_tokens_per_section=10_000,
    ),
    output_dir=os.environ.get("CAPSULES_OUTPUT_DIR", "."),
    
    # stage 2 generate data
    num_samples=1,
    batch_size=1,
    max_num_batches_in_parallel=1,

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


# configs.append(config_compare)
# configs.append(config_preceding)
# configs.append(config_cheat)
# configs.append(config_next)
# configs.append(config_modify)

# configs.append(config_simpleqa_history)
# configs.append(config_historyqa)
# configs.append(config_recallqa)

configs.append(config_direct)
configs.append(config_simpleqa)
# configs.append(config_attempt)


if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])

