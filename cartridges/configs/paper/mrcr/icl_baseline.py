import os
from pathlib import Path
import pydrantic

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.train import GenerateDatasetConfig
from capsules.utils import WandBConfig
from capsules.config import HFModelConfig
from capsules.datasets import ( CapsuleDatasetLatest )

from capsules.tasks.mrcr import ( MRCREvalDataset, MRCRGenerateDataset ) 
from capsules.tasks.mrcr import MRCRSectionedContextConfig



file_name = Path(__file__).stem

model_path = "meta-llama/Llama-3.2-3B-Instruct"

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the information below to answer the questions.

<reference>
{{content}}
</reference>
"""


configs = []
bs = 1
document_id = -1

    
for num_tokens, lr in [(131702, 0.00)]:
    configs.append(
        TrainConfig(
            name=f"mrcr_{file_name}_doc{document_id}_nt{num_tokens}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),
            dataset=CapsuleDatasetLatest.Config(
                data_sources=[("/data/simran/2025-05-15-20-56-17-generate/a506c41b-9ac4-4267-92a6-5198fdaed844/artifact/dataset.pkl", None),], # Garbage
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),


            context=MRCRSectionedContextConfig(
                document_id=document_id, 
            ),

            # Evals and generations
            generate_every_n_steps=512,
            generate_datasets=[
                GenerateDatasetConfig(
                    name_for_wandb="generate-mrcr",
                    dataset=MRCRGenerateDataset.Config(
                        document_id=document_id,
                        use_cot=True,
                    ),
                )
            ],

            eval_every_n_steps=256,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb="eval-mrcr",
                    local_batch_size=8,
                    dataset=MRCREvalDataset.Config(
                        document_id=document_id,
                        use_cot=True,
                    ),
                )
            ],


            generate_max_new_tokens=512,
            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=num_tokens
            ),
            
            loss_type="logits",
            save_every_n_steps=128,
            epochs=0,
            lr=lr,
            save_to_wandb = False,
            wandb=WandBConfig(
                project="capsules",
                tags=["mrcr", "icl", f"doc{document_id}"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=1,
        )
    )

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])


