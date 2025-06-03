import os
from pathlib import Path
import pydrantic

from cartridges.initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from cartridges.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import ( CartridgeDatasetLatest )
from cartridges.utils import WandBConfig
from cartridges.tasks.mrcr import ( MRCREvalDataset, MRCRGenerateDataset ) 
from cartridges.tasks.mrcr import MRCRSectionedContextConfig


file_name = Path(__file__).stem
bs = 64
configs = []



document_id = -1

for num_tokens, lr in [(2048, 0.05)]:

    configs.append(
        TrainConfig(
            # Data and initialization
            name=f"mrcr_{file_name}_nt{num_tokens}_auto_lr{lr}_{document_id}_dataset3",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),

            tokenizer="meta-llama/Llama-3.2-3B-Instruct",
            lr=lr,
            dataset=CartridgeDatasetLatest.Config(
                data_sources=[
                    # ("/data/simran/2025-05-15-20-56-17-generate/a506c41b-9ac4-4267-92a6-5198fdaed844/artifact/dataset.pkl", None),
                    # ("/data/simran/2025-05-15-21-27-47-generate/f33382f8-f6f2-4dd0-90e7-74725cc6b77b/artifact/dataset.pkl", None),
                    # ("/data/simran/2025-05-15-23-10-53-generate/340f5f77-f8df-428f-8cbf-0bdb666e3190/artifact/dataset.pkl", None),
                    ("/data/simran/2025-05-15-23-47-22-generate_on_boundaries/31741323-9643-422b-a6a3-6e1998a7a624/artifact/dataset.pkl", None),
                ],
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),

            context=MRCRSectionedContextConfig(
                document_id=document_id, 
            ),

            generate_max_new_tokens=512,
            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=num_tokens
            ),

            # Evals and generations
            generate_every_n_steps=128,
            generate_datasets=[
                GenerateDatasetConfig(
                    name_for_wandb="generate-mrcr",
                    dataset=MRCRGenerateDataset.Config(
                        document_id=document_id,
                    ),
                )
            ],

            eval_every_n_steps=128,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb="eval-mrcr",
                    local_batch_size=8,
                    dataset=MRCREvalDataset.Config(
                        document_id=document_id,
                    ),
                )
            ],

            # Training setup 
            loss_type="logits",
            save_every_n_steps=256,
            epochs=1,
            
            wandb=WandBConfig(
                project="cartridges",
                tags=["mrcr", "auto_train", f"doc{document_id}"],
                entity="hazy-research",
            ),

            output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=4,
            distributed_backend="gloo",
        )
    )


if __name__ == "__main__":
   for config in configs:
       pydrantic.main([config])


