import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.tasks.finance import FinanceBenchEvalDataset, FinanceBenchGenerateDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
DOC_NAME = "AMD_2022_10K"

for num_tokens in [8192]:
    config = TrainConfig(
        name=f"{file_name}_{DOC_NAME}_nt{num_tokens}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        dataset=CapsuleDataset.Config(
            data_sources=[
                (f"hazy-research/capsules/generate_amd_2022_10k_higher_temp:v0", None),
            ],  
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        generate_every_n_steps=512,
        generate_datasets=[
            GenerateDatasetConfig(
                name_for_wandb="finance-bench",
                dataset=FinanceBenchGenerateDataset.Config(
                    doc_names=[DOC_NAME],
                )
            ),
        ],
        eval_every_n_steps=16,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb="finance-ppl",
                local_batch_size=16,
                dataset=FinanceBenchEvalDataset.Config(
                    doc_names=[DOC_NAME],
                    cot=False,
                    label_type="tokens",
                    data_sources=[]  # ignore this arg

                ), 
                only_eval_rank_0=True,
            ),
        ],
        generate_max_new_tokens=1024,
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens,
            num_frozen_tokens=1,
        ),
        loss_type="logits",
        save_every_n_steps=64,
        epochs=1,
        lr=5e-3,
        wandb=WandBConfig(
            project="capsules",
            tags=["cache_tuning", "development"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        global_batch_size=32,
        local_batch_size=4,
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)

