import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable
from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.optim import CosWithWarmup
from capsules.tasks.swde import SWDEEvalDataset, SWDEMultipleChoiceGenerateDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDatasetLatest
from capsules.utils import WandBConfig


file_name = Path(__file__).stem

configs = []
bs = 64
configs = []
num_tokens = 8192

prefix = "hazy-research/capsules"
expt_tag = "simpleqa_no_thoughts"

simpleqa_mappings = {
    "0000.htm": "/data/simran/capsules/outputs/2025-04-23-13-25-54-m04d23_simpleqa_no_thoughts/c22035a8-9a6b-4065-9853-0c09408bec23/artifact/dataset.pkl",
}


configs = []

for html_page in ['0000.htm']: 

    html_data_path = simpleqa_mappings[html_page]

    for lr in [8e-3]: #1e-3, 5e-4, 5e-3

        config = TrainConfig(
                name=FormatStringVariable(f"{file_name}_book{html_page}_lr{lr}"),
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
                    data_sources=[(html_data_path, None),],
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
                        dataset=SWDEMultipleChoiceGenerateDataset.Config(
                            webpage_id=html_page, 
                            cot=False,
                        ),
                        name_for_wandb=f"swde_imdb_{expt_tag}_{html_page}_mc",
                    ),
                ],

                # evaluation
                eval_every_n_steps=64,
                eval_datasets=[
                    EvalDatasetConfig(
                        name_for_wandb=f"swde_imdb_{expt_tag}_mc",
                        local_batch_size=16,
                        dataset=SWDEEvalDataset.Config(
                            webpage_id=html_page,
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
                epochs=1,

                # wandb
                wandb=WandBConfig(
                    project="capsules",
                    tags=["train", f"swde_imdb_{expt_tag}", f"webpage{html_page}"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
                global_batch_size=bs,
                local_batch_size=4,
            )
        configs.append(config)
        
        print(f"Added config: {file_name} - {html_page} - {lr}")

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)



