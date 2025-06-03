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


configs = []
parallel = True
expt_tag = ""
document_id = -2


# directly generate training data
config_direct = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_n{{num_samples}}_direct{expt_tag}_doc{document_id}"),
    convo_generator=DirectGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=10,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=document_id,
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
        tags=[f"mrcr", "generate", f"mrcr_doc{document_id}"],
    ),
)

# simple qa generator
config_simpleqa = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_n{{num_samples}}_simpleqa{expt_tag}_doc{document_id}_variant"),
    convo_generator=SimpleQAGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=512,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=document_id,
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
        tags=[f"mrcr", "generate", f"mrcr_doc{document_id}"],
    ),
)

# attempt
config_attempt = GenerateTrainingConfig(
    name=FormatStringVariable(f"{file_name}_n{{num_samples}}_attempt{expt_tag}_doc{document_id}"),
    convo_generator=CategoryGenerator.Config(
        client=client,
        temperature=0.2,
        max_completion_tokens=1024,
        prompt_template="",
        system_prompt_template="",
    ),
    context=MRCRContextConfig(
        document_id=document_id,
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
        tags=[f"mrcr", "generate", f"mrcr_doc{document_id}"],
    ),
)



configs.append(config_attempt)
configs.append(config_direct)
configs.append(config_simpleqa)


if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])

