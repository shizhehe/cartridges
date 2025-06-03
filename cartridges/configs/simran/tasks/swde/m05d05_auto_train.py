import os
from pathlib import Path
import pydrantic

from capsules.optim import CosWithWarmup
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.tasks.swde import ( SWDEMultipleChoiceGenerateDataset, SWDEEvalDataset )
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from capsules.config import HFModelConfig
from capsules.datasets import ( CapsuleDatasetLatest )
from capsules.utils import WandBConfig


file_name = Path(__file__).stem

configs = []
bs = 64


# Experiment: Cartridge V1 SimplePromptSampler
dataset_sources = [
    # ("/data/simran/capsules/outputs/2025-05-05-20-38-56-m05d05_auto_generate/dda1a45c-2921-435e-babe-625e4f850d9d/artifact/dataset.pkl", None),
    # ("/data/simran/capsules/outputs/2025-05-05-21-31-48-m05d05_auto_generate/20012b21-cc2e-4b2a-a450-5bd2b0d42001/artifact/dataset.pkl", None),
    # ("/data/simran/capsules/outputs/2025-05-05-22-03-39-m05d05_auto_generate/d74baeb0-9df3-4bf3-ae1a-c9cd391281e5/artifact/dataset.pkl", None),
    ("/data/simran/capsules/outputs/2025-05-05-23-04-02-m05d05_auto_generate/0a75e6af-7ef5-482a-b1c2-f8b05f6dcced/artifact/dataset.pkl", None),
]
dataset_mix = "sampler"

html_page = '0000.htm'
configs = []


for num_tokens, lr in [(4096, 0.05), (4096, 0.005), (4096, 0.008)]: # 
    configs.append(

        TrainConfig(
            name=f"{file_name}_doc{html_page}_nt{num_tokens}_auto_data{dataset_mix}_lr{lr}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),

            lr=lr,
            lr_scheduler=CosWithWarmup.Config(
                warmup_steps=128,
                max_steps=2048,
                warmup_min_lr=5e-3,
                alpha_f=0.1,
            ),


            dataset=CapsuleDatasetLatest.Config(
                data_sources=dataset_sources,
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),

            generate_every_n_steps=64,
            generate_datasets=[
                GenerateDatasetConfig(
                    dataset=SWDEMultipleChoiceGenerateDataset.Config(
                        webpage_id=html_page, 
                        cot=False,
                    ),
                    name_for_wandb=f"swde_imdb_{html_page}_mc",
                ),
            ],

            eval_every_n_steps=64,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb=f"swde_imdb_mc",
                    local_batch_size=16,
                    dataset=SWDEEvalDataset.Config(
                        webpage_id=html_page,
                        max_questions=256,
                        label_type="tokens",
                        data_sources=[]  # ignore this arg
                    )
                )
            ],

            generate_max_new_tokens=512,
            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=num_tokens
            ),

            loss_type="logits",
            save_every_n_steps=128,
            epochs=1,
            
            wandb=WandBConfig(
                project="capsules",
                tags=["swde", "auto_train"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=4,
        )
    )

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])


