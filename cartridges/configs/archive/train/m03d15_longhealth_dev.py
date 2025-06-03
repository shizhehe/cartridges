import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.kv_initialization.strategies.prompt import PromptInitializer
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleGenerateDataset
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem
configs = []
for num_tokens in [1024]:
    config = TrainConfig(
        name=f"{file_name}_nt{num_tokens}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        dataset=CapsuleDataset.Config(
            data_sources=[
                # ("hazy-research/capsules/m03d12_longhealth_basic_qa_train:v8", None),

                # (n=8192) added prompt that forces the model to mention identifiers
                ("hazy-research/capsules/m03d12_longhealth_basic_qa_train:v16", None),
            ],
            is_wandb=True,
            label_type="tokens",
        ),
        generate_every_n_steps=64,
        generate_datasets=[
            GenerateDatasetConfig(
                name_for_wandb="multiple_choice_generations",
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=None,
                    max_questions=128,
                )
            ),
        ],
        eval_every_n_steps=16,
        eval_datasets=[
            # EvalDatasetConfig(
            #     name_for_wandb="longhealth_multiple_choice",
            #     batch_size=16,
            #     dataset=LongHealthEvalDataset.Config(
            #         patient_ids=None,
            #         max_questions=256,
            #         label_type="tokens",
            #         data_sources=[]  # ignore this arg
            #     )
            # ),
            EvalDatasetConfig(
                name_for_wandb="generated_questions",
                local_batch_size=16,
                dataset=CapsuleDataset.Config(
                    data_sources=[
                        ("hazy-research/capsules/m03d12_longhealth_basic_qa_test:v0", None),
                    ],
                    is_wandb=True,
                    label_type="tokens",
                ),
            )
        ],
        generate_max_new_tokens=512,
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=16384
        ),
        loss_type="tokens",
        save_every_n_steps=64,
        epochs=1,
        lr=5e-3,
        wandb=WandBConfig(
            project="capsules",
            tags=["cache_tuning", "development"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        train_batch_size=2,
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
