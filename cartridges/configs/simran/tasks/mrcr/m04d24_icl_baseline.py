import os
from pathlib import Path

import pydrantic

from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.clients.tokasaurus import TokasaurusClient
from capsules.tasks.mrcr import MRCRContextConfig, MRCRGenerateDataset, MRCRGenerateDatasetTask2, MRCRGenerateDatasetTask3
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

file_name = Path(__file__).stem

configs = []
SYSTEM_PROMPT_TEMPLATE = f"""Please reference the conversation history below to answer the questions.

<conversation>
{{content}}
</conversation>
"""

document_id = -2
max_tokens = 1024
configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{document_id}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            max_completion_tokens=max_tokens
        ),
        dataset=GenerateDatasetConfig(
            dataset=MRCRGenerateDataset.Config(
                document_id=document_id, 
                cot=False,
            ),
            name_for_wandb=f"mrcr_generate_doc{document_id}",
        ),
        context=MRCRContextConfig(
            document_id=document_id,
            max_tokens_per_section=-1,
        ),
        # generate_max_new_tokens=512,
        max_num_batches_in_parallel=1,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"mrcr", "genbaseline", f"conversation_{document_id}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

configs.append(
    GenerateBaselineConfig(
        name=f"{file_name}_{document_id}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            max_completion_tokens=max_tokens
        ),
        dataset=GenerateDatasetConfig(
            dataset=MRCRGenerateDatasetTask2.Config(
                document_id=document_id, 
                cot=False,
            ),
            name_for_wandb=f"mrcr_icl_mc_task2",
        ),
        context=MRCRContextConfig(
            document_id=document_id,
            max_tokens_per_section=-1,
        ),
        # generate_max_new_tokens=512,
        max_num_batches_in_parallel=1,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"mrcr", "genbaseline", f"conversation_{document_id}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
)

configs.append(
    GenerateBaselineConfig(
        name=f"{file_name}_{document_id}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            max_completion_tokens=max_tokens
        ),
        dataset=GenerateDatasetConfig(
            dataset=MRCRGenerateDatasetTask3.Config(
                document_id=document_id, 
                cot=False,
            ),
            name_for_wandb=f"mrcr_icl_mc_task3",
        ),
        context=MRCRContextConfig(
            document_id=document_id,
            max_tokens_per_section=-1,
        ),
        # generate_max_new_tokens=512,
        max_num_batches_in_parallel=1,
        batch_size=16,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"mrcr", "genbaseline", f"conversation_{document_id}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
)

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])

