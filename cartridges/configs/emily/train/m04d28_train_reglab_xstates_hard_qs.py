import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.optim import CosWithWarmup
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.reglab import ReglabHousingCategoriesQAGenerateDataset, ReglabHousingQAGenerateDataset
from capsules.tasks.reglab.utils import ALL_STATES
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


NUM_STATES = 5
states = ALL_STATES[:NUM_STATES]

if NUM_STATES == 5:
    data_sources = [
        # ("/data/sabri/code/capsules/output/2025-04-24-11-57-31-m04d24_generate_reglab_ir/940a9b6d-531c-4ab6-b651-6f7a1aecf7f1/artifact/dataset.pkl", None),
        # ("/data/sabri/code/capsules/output/2025-04-24-18-14-57-m04d24_generate_reglab_ir/1c97f93f-c033-4171-8ed9-d087c8ff18de/artifact/dataset.pkl", None),
        # ("/data/sabri/code/capsules/output/2025-04-24-18-28-47-m04d24_generate_reglab_ir/46ecfde5-ad8b-488d-8aff-fc167a056ab8/artifact/dataset.pkl", None),


        # ("/data/sabri/code/capsules/output/2025-04-25-15-14-56-m04d24_generate_reglab_ir/d00311b2-044b-478d-9c17-5644b22798cc/artifact/dataset.pkl", None),
        # ("/data/sabri/code/capsules/output/2025-04-25-15-42-46-m04d24_generate_reglab_ir/c99296b8-6f75-4263-bb59-910a77daf381/artifact/dataset.pkl", None),
        # ("/data/sabri/code/capsules/output/2025-04-25-15-50-10-m04d24_generate_reglab_ir/2333997e-55eb-4dc9-811d-496aad654a8f/artifact/dataset.pkl", None),
        # ("/home/emilyryliu/capsules-outputs/2025-04-28-22-12-46-m04d28_generate_reglab_hard_no_ex/24e5167e-f51b-4056-b683-1e1c71b02937/artifact/dataset.pkl", None),
        # ("/home/emilyryliu/capsules-outputs/2025-04-28-22-18-34-m04d28_generate_reglab_creative_no_ex/c6ea504c-52c9-4b30-a409-e6eef36a6fe3/artifact/dataset.pkl", None),
        ("hazy-research/capsules/m04d28_generate_reglab_hard_no_ex_8192:v1", None),
    ]
elif NUM_STATES == 20:
    data_sources = [
        ("/data/sabri/code/capsules/output/2025-04-24-10-05-18-m04d24_generate_reglab_ir/3d701f20-c8e7-4756-b067-7079da867f1f/artifact/dataset.pkl", None),
    ]
else:
    raise ValueError(f"Invalid number of states: {NUM_STATES}")


configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_states{NUM_STATES}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        
        lr=lr,

        dataset=CapsuleDatasetLatest.Config(
            data_sources=data_sources,
            max_sequence_length=1024,
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        
        save_every_n_steps=128,
        generate_every_n_steps=128,
        generate_max_new_tokens=512,
        generate_datasets=[
            GenerateDatasetConfig(
                dataset=ReglabHousingCategoriesQAGenerateDataset.Config(
                    states = states,
                    cot=True,
                ),
                name_for_wandb=f"reglab_categories_generate",
                num_samples=1,
                temperature=0.3
            ),
        ],
        eval_every_n_steps=64,
        eval_datasets=[
            # EvalDatasetConfig(
            #     name_for_wandb="longhealth_mc",
            #     local_batch_size=16,
            #     dataset=LongHealthEvalDataset.Config(
            #         patient_ids=patient_ids,
            #         max_questions=256,
            #         label_type="tokens",
            #         data_sources=[]  # ignore this arg
            #     )
            # )
        ],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        epochs=1,

        wandb=WandBConfig(
            project="capsules",
            tags=["train", "reglab-housing", f"{NUM_STATES}states"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=4,
    )
    for lr in [5e-3] #, 6e-3, 5e-3]
    for num_tokens in [8192] #2048, 4096, 6144]
]

if __name__ == "__main__":
    pydrantic.main(configs)
