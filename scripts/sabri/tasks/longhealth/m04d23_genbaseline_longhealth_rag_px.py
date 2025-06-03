import os
from pathlib import Path

import pydrantic

from cartridges.generate_baseline import GenerateBaselineConfig, ICLBaseline, RAGBaseline
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.retrievers import OpenAIRetriever
from cartridges.tasks.longhealth import LongHealthContextConfig, LongHealthMultipleChoiceGenerateDataset
from cartridges.train import GenerateDatasetConfig

from cartridges.utils import WandBConfig

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)


file_name = Path(__file__).stem

NUM_PATIENTS = 10

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the patient's medical records included below to answer the user's questions.

<patient-records>
{{content}}
</patient-records>
"""

patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]
configs = [
    GenerateBaselineConfig(
        name=f"{file_name}_{patients_str}_topk{top_k}",
        generator=RAGBaseline.Config(
            client=tokasaurus_client,
            system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            retriever=OpenAIRetriever.Config(
                embedding_model="text-embedding-3-large",
                max_tokens_per_chunk=256
            ),
            top_k=top_k,
            temperature=0.3,
        ),
        dataset=GenerateDatasetConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids, 
                cot=True,
            ),
            name_for_wandb=f"longhealth_mc",
            num_samples=32,
            temperature=0.3,
        ),
        context=LongHealthContextConfig(
            patient_ids=patient_ids
        ),
        generate_max_new_tokens=512,
        max_num_batches_in_parallel=32,
        batch_size=32,
        wandb=WandBConfig(
            project="cartridges",
            tags=[f"longhealth_genbaseline_{patients_str}", "eval"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    )
    for top_k in [
        # 1, 2, 4, 
        # 8, 16, 32
        64
    ]
]
if __name__ == "__main__":
    pydrantic.main(configs)
