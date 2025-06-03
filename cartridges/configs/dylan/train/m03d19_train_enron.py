import os
from pathlib import Path

import pydrantic

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
# for num_tokens in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:

for num_tokens in [2048, 4096, 8192]:
    config = TrainConfig(
        name=f"{file_name}_nt{num_tokens}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        dataset=CapsuleDataset.Config(
            data_sources=[

                # FIXME: this is the wrong dataset, have it point to the enron dataset you generated
                ("hazy-research/capsules/m03d17_enron_basic_qa_train:v1", None),
            ],  
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        generate_every_n_steps=64,
        generate_datasets=[
            # SKIP for now
            # GenerateDatasetConfig(
            #     name_for_wandb="multiple_choice_generations",
            #     dataset=LongHealthMultipleChoiceGenerateDataset.Config(
            #         patient_ids=[patient_id],
            #         max_questions=128,
            #         cot=True
            #     )
            # ),
        ],
        eval_every_n_steps=64,
        eval_datasets=[
            # EvalDatasetConfig(
            #     name_for_wandb="longhealth_multiple_choice",
            #     local_batch_size=16,
            #     dataset=LongHealthEvalDataset.Config(
            #         patient_ids=[patient_id],
            #         max_questions=256,
            #         label_type="tokens",
            #         data_sources=[]  # ignore this arg
            #     )
            # ),
            # FIXME: this is the wrong dataset, have it point to the enron dataset you generated
            # you need to generate a test dataset using GPT-4o
            EvalDatasetConfig(
                name_for_wandb="generated_questions",
                local_batch_size=16,
                dataset=CapsuleDataset.Config(
                    data_sources=[
                        ("hazy-research/capsules/m04d03_enron_basic_qa_test:v0", None),
                    ],
                    is_wandb=True,
                    label_type="tokens",
                ),
            )
        ],
        generate_max_new_tokens=1024,
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
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
        global_batch_size=256,
        local_batch_size=4,
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
