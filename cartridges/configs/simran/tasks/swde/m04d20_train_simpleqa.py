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

html_page = '0349.htm'
prefix = "hazy-research/capsules"
expt_tag = "simpleqa_origqa_v3"

simpleqa_mappings = {
    "0349.htm": "/data/simran/capsules/outputs/2025-04-21-14-43-21-m04d21_simpleqa_v3/38fc113d-a8fc-4cb4-bf42-5c9374a8920d/artifact/dataset.pkl",
    "0000.htm": "/data/simran/capsules/outputs/2025-04-21-17-03-06-m04d21_simpleqa_v3/12c4cd91-44ef-4996-bb2e-e806c17409d5/artifact/dataset.pkl",
}


configs = []

for lr in [8e-3]: #1e-3, 5e-4, 5e-3

    configs.append(

        TrainConfig(
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
                data_sources=[
                    
                    # simpleqa attempt #1
                    # original simpleqa 
                    # (f"{prefix}/m04d20_simpleqa_0349.htm:v0", None),

                    # simpleqa attempt #2 (best one so far)
                    # better formatting and larger contexts
                    # ("/data/simran/capsules/outputs/2025-04-21-10-39-53-m04d20_simpleqa_v2/e08f8859-5a7d-48ff-a81a-0faccaab91e8/artifact/dataset.pkl", None),

                    # simpleqa attempt #3
                    # better formatting and larger variation in context length
                    ("/data/simran/capsules/outputs/2025-04-21-14-43-21-m04d21_simpleqa_v3/38fc113d-a8fc-4cb4-bf42-5c9374a8920d/artifact/dataset.pkl", None),

                    
                    ##################################### 

                    # memorize attempt #1
                    # (f"{prefix}/m04d20_memorize_0349.htm_memorize:v0", None),

                    # memorize attempt #2
                    # (f"{prefix}/m04d20_memorize_v1_0349.htm_memorize_v1:v0", None),

                    # memorize attempt #3
                    # listy summary for memorization
                    # (f"{prefix}/m04d20_memorize_v2_0349.htm_memorize_v2:v7", None),

                    # memorize attempt #4
                    # listy summary for memorization
                    # ("/data/simran/capsules/outputs/2025-04-20-22-10-51-m04d20_memorize_v2/c36370b8-919f-4862-9a3a-2a755ce440dc/artifact/dataset.pkl", None),

                    # memorize attempt #5
                    # paragraph in summary for memorization
                    # (f"/data/simran/capsules/outputs/2025-04-21-09-36-31-m04d20_memorize_v2/586a2060-2f06-4132-bedf-040209cd3e25/artifact/dataset.pkl", None),
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
    )


if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)



