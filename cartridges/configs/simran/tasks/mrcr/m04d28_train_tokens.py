import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable
from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.optim import CosWithWarmup
from capsules.tasks.mrcr import MRCREvalDataset, MRCRGenerateDataset, MRCRGenerateDatasetTask2, MRCRGenerateDatasetTask3
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleDatasetLatest
from capsules.utils import WandBConfig


file_name = Path(__file__).stem

configs = []
gbs = 32
lbs = 1
configs = []
num_tokens = 2048

document_id = -2

prefix = "hazy-research/capsules"
expt_tag = f"mrcr_direct_only_{num_tokens}"


configs = []
# data_paths = [
#     "/data/simran/m05d01_fair_reformatter_0_n32768_simpleqa_variant/artifact/dataset.pkl",
#     "/data/simran/m05d01_fair_reformatter_0_n32768_direct/artifact/dataset.pkl",
#     "/data/simran/m05d01_fair_reformatter_0_n1_attempt/artifact/dataset.pkl",
# ]


data_paths = [
    "/data/simran/m05d01_fair_reformatter_n32768_simpleqa_doc-2_variant/artifact/dataset.pkl",
    "/data/simran/m05d01_fair_reformatter_n32768_direct_doc-2/artifact/dataset.pkl",
    "/data/simran/m05d01_fair_reformatter_n1_attempt_doc-2/artifact/dataset.pkl",
]


for lr in [8e-3, 5e-3]:  
    config = TrainConfig(
            name=FormatStringVariable(f"{file_name}_lr{lr}_{expt_tag}_tokens{num_tokens}_doc{document_id}"),
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),
            
            lr=lr,
            lr_scheduler=CosWithWarmup.Config(
                warmup_steps=128,
                max_steps=2048,
                warmup_min_lr=5e-5,
                alpha_f=0.1,
            ),
            dataset=CapsuleDatasetLatest.Config(
                data_sources=[
                    (path, None)
                    for path in data_paths
                ],
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),
            save_every_n_steps=2048,


            # generation
            generate_every_n_steps=64,
            generate_max_new_tokens=512,
            generate_datasets=[
                GenerateDatasetConfig(
                    dataset=MRCRGenerateDataset.Config(
                        document_id=document_id, 
                        cot=False,
                    ),
                    name_for_wandb=f"mrcr_{expt_tag}_mc",
                ),
                GenerateDatasetConfig(
                    dataset=MRCRGenerateDatasetTask2.Config(
                        document_id=document_id, 
                        cot=False,
                    ),
                    name_for_wandb=f"mrcr_{expt_tag}_mc_task2",
                ),
                GenerateDatasetConfig(
                    dataset=MRCRGenerateDatasetTask3.Config(
                        document_id=document_id, 
                        cot=False,
                    ),
                    name_for_wandb=f"mrcr_{expt_tag}_mc_task3",
                ),
            ],

            # evaluation
            eval_every_n_steps=64,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb=f"mrcr_{expt_tag}_mc",
                    local_batch_size=16,
                    dataset=MRCREvalDataset.Config(
                        document_id=document_id, 
                        max_questions=256,
                        label_type="tokens",
                        data_sources=[]  # ignore this arg
                    )
                )
            ],

            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=num_tokens
            ),
            loss_type="logits",
            epochs=48,

            # wandb
            wandb=WandBConfig(
                project="capsules",
                tags=["train", f"mrcr_imdb_{expt_tag}_doc{document_id}"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=gbs,
            local_batch_size=lbs,
        )
    configs.append(config)
    
    print(f"Added config: {file_name} - {lr}")

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)



