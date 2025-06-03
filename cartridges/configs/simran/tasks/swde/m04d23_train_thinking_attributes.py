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
# expt_tag = "simpleqaold_fairreformat_mix"
expt_tag = "fairreformat_only"

simpleqa_mappings = {
    "0000.htm": [
        # "/data/simran/capsules/outputs/2025-04-23-14-53-32-m04d23_think_attributes/df8c27d2-5aa2-4439-9b87-a42ae74e0f8a/artifact/dataset.pkl",
        # "/data/simran/capsules/outputs/2025-04-23-17-33-02-m04d23_jsonqa/9f9286ad-dd2f-4b3b-8591-07ec33b12eb3/artifact/dataset.pkl",
        # "/data/simran/capsules/outputs/2025-04-21-17-03-06-m04d21_simpleqa_v3/12c4cd91-44ef-4996-bb2e-e806c17409d5/artifact/dataset.pkl",

        # new ones
        # "/data/simran/capsules/outputs/2025-04-24-11-57-23-m04d21_simpleqa_v3/46cc8155-cca7-4537-970b-a1929f6e4a68/artifact/dataset.pkl"
        # "/data/simran/capsules/outputs/2025-04-24-14-21-09-m04d24_multiqa/77794afb-34ed-435a-a977-86637efdd3e6/artifact/dataset_clustered.pkl",


        # "/data/simran/capsules/outputs/2025-04-24-17-53-10-m04d24_multiqa/a2bbd270-08ac-4da9-829c-75bf3d2ec85c/artifact/dataset.pkl"
        # "/data/simran/capsules/outputs/2025-04-24-17-58-13-m04d24_multiqa/787ddf2b-c2c2-4972-8b3e-8c1ec0b6dd0c/artifact/dataset.pkl"

        # "/data/simran/capsules/outputs/2025-04-24-18-08-53-m04d24_multiqa/364fd5fd-b42e-4a2e-94af-64f35e856702/artifact/dataset.pkl"
        # "/data/simran/capsules/outputs/2025-04-24-20-26-16-m04d21_simpleqa_v3/0d393daf-1796-483e-8b3a-b27797b13141/artifact/dataset.pkl",
        # "/data/simran/capsules/outputs/2025-04-24-20-50-37-m04d24_reformatqa/37f0f00e-5c16-4ec0-b303-d119707ed15f/artifact/dataset.pkl"

        # fairer reformatter
        "/data/simran/capsules/outputs/2025-04-25-14-43-56-m04d24_reformatqa/ad7c49cc-e4e0-4b56-b8a7-67c2c594c059/artifact/dataset.pkl"
    ],

    # "0001.htm": [
    #     "/data/simran/capsules/outputs/2025-04-21-17-03-06-m04d21_simpleqa_v3/1491fc09-1f61-4cba-a5c4-97ec42ca0028/artifact/dataset.pkl",
    #     "/data/simran/capsules/outputs/2025-04-23-23-00-48-m04d23_jsonqa/bcb8a33e-c590-44b5-9f5c-e31603f2fa2e/artifact/dataset.pkl"
    # ],

    # "0002.htm": [
    #     "/data/simran/capsules/outputs/2025-04-21-17-03-06-m04d21_simpleqa_v3/06116af2-852d-47c7-86a0-35469668461e/artifact/dataset.pkl",
    #     "/data/simran/capsules/outputs/2025-04-23-23-00-48-m04d23_jsonqa/97e7b505-e4b3-42cc-88a5-3005121414f8/artifact/dataset.pkl"
    # ],

    # "0349.htm": [
    #     "/data/simran/capsules/outputs/2025-04-23-23-00-48-m04d23_jsonqa/14ff5c83-de7a-492f-b316-d645d62debab/artifact/dataset.pkl",    
    # ],
}


configs = []

for html_page in ['0000.htm']: # '0000.htm', '0001.htm', '0002.htm'

    html_data_paths = simpleqa_mappings[html_page]

    for lr in [8e-3, 5e-3]: #1e-3, 5e-4, 5e-3, 8e-3,  

        config = TrainConfig(
                name=FormatStringVariable(f"{file_name}_book{html_page}_lr{lr}_{expt_tag}"),
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
                        (html_data_path, None)
                        for html_data_path in html_data_paths
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
        configs.append(config)
        
        print(f"Added config: {file_name} - {html_page} - {lr}")

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main(configs)



