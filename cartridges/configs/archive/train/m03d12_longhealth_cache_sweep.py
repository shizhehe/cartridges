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
for max_tokens in [128, 256, 512, 1_024, 2_048, 4_096, 8_192, 16_384]:
    config = TrainConfig(
        name=f"{file_name}_mt{max_tokens}",
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
        generate_max_new_tokens=64,
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=max_tokens
        ),
        loss_type="tokens",
        save_every_n_steps=512,
        epochs=1,
        lr=5e-3,
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
