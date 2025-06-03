import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.kv_initialization.strategies.prompt import PromptInitializer
from capsules.train import TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem
configs = []
for num_tokens in [1024, 2048, 4096]:
    for lr in [1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 5e-3, 1e-2, 3e-2, 5e-2]:
        config = TrainConfig(
            name=f"{file_name}_lr{lr}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),
            dataset=CapsuleDataset.Config(
                data_sources=[
                    ("hazy-research/capsules/m03d12_longhealth_basic_qa_train:v8", None),
                ],
                is_wandb=True,
                label_type="tokens",
            ),
            generate_every_n_steps=64,
            generate_dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=None,
                max_questions=32,
            ),
            eval_every_n_steps=64,
            eval_dataset=LongHealthEvalDataset.Config(
                patient_ids=None,
                max_questions=256,
                label_type="tokens",
                data_sources=[]  # ignore this arg
            ),
            generate_max_new_tokens=256,
            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=2048
            ),
            loss_type="tokens",
            save_every_n_steps=512,
            epochs=1,
            lr=lr,
            wandb=WandBConfig(
                project="capsules",
                tags=["cache_tuning", "development"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            batch_size=4,
        )
        configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
