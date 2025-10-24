import os
from pathlib import Path

import pydrantic

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.longhealth.context import LongHealthStructuredContextConfig
from capsules.tasks.mmlu import MMLUGenerateDataset
from capsules.tasks.qasper.context import QasperStructuredContextConfig
from capsules.train import GenerateDatasetConfig

from capsules.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    # url="https://hazyresearch--tksrs-add-top-lo-llama-3b-1xh100-min0-serve.modal.run/v1",
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


file_name = Path(__file__).stem

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the patient medical records included below to answer the user's questions.

<patient-records>
{{content}}
</patient-records>
"""

# SYSTEM_PROMPT_TEMPLATE = "Please answer the user's question."

TOPIC = "question"

configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{TOPIC}",
        generator=ICLBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            temperature=0.0,
            max_completion_tokens=512,
        ),
        dataset=GenerateDatasetConfig(
            dataset=MMLUGenerateDataset.Config(
                num_problems=512,
            ),
            batch_size=64,
            name_for_wandb=f"mmlu"
        ),
        context=QasperStructuredContextConfig(topic=TOPIC),
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(
            project="capsules",
            tags=[f"qasper", "genbaseline", f"qasper_{TOPIC}", "icl"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
    )
]

if __name__ == "__main__":
    pydrantic.main(configs)
