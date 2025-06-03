import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset
from cartridges.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
# for num_tokens in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
for num_tokens in [2048]:
    lr = 5e-3
    config = TrainConfig(
        name=f"{file_name}_ntokens={num_tokens}_lr{lr:.0e}",
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            # pretrained_model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        ),
        dataset=CartridgeDataset.Config(
            data_sources=[
                # ("hazy-research/Cartridges/m03d12_longhealth_basic_qa_train:v8", None),

                # (n=8192) added prompt that forces the model to mention identifiers
                # ("hazy-research/Cartridges/m03d12_longhealth_basic_qa_train:v19", None),

                # (n=32768) chunked answers
                # ("hazy-research/Cartridges/m03d17_generate_longhealth_p01:v0", None),

                # (n=32768) with full context answers
                # ("hazy-research/Cartridges/m03d18_generate_longhealth_p01_full_context:v1", None),

                # (n=32768) with full context answers and cot encouragement
                # ("hazy-research/Cartridges/m03d19_generate_longhealth_sweep_patients_p01_32k:v0", None),

                # (n=8192) with full context answers and cot encouragement
                ("hazy-research/Cartridges/m03d19_generate_longhealth_sweep_patients_p01:v0", None),

            ],  
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        generate_every_n_steps=16,
        generate_datasets=[
            # GenerateDatasetConfig(
            #     name_for_wandb="multiple_choice_generations",
            #     dataset=LongHealthMultipleChoiceGenerateDataset.Config(
            #         patient_ids=["patient_01"],
            #         max_questions=128,
            #         cot=True
            #     )
            # ),
        ],
        eval_every_n_steps=16,
        eval_datasets=[
            # EvalDatasetConfig(
            #     name_for_wandb="longhealth_multiple_choice",
            #     local_batch_size=16,
            #     dataset=LongHealthEvalDataset.Config(
            #         patient_ids=["patient_01"],
            #         max_questions=256,
            #         label_type="tokens",
            #         data_sources=[]  # ignore this arg
            #     )
            # ),
            EvalDatasetConfig(
                name_for_wandb="generated_questions",
                local_batch_size=16,
                dataset=CartridgeDataset.Config(
                    data_sources=[
                        ("hazy-research/Cartridges/m03d12_longhealth_p01_basic_qa_test:v0", 4),
                    ],
                    is_wandb=True,
                    label_type="tokens",
                ),
            )
        ],
        generate_max_new_tokens=1024,
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens,
            num_frozen_tokens=1,
        ),
        loss_type="logits",
        save_every_n_steps=64,
        epochs=1,
        lr=5e-4,
        wandb=WandBConfig(
            project="cartridges",
            tags=["cache_tuning", "development"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        global_batch_size=16,
        local_batch_size=4,
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
