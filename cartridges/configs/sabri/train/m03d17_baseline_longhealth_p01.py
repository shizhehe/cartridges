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
for num_tokens in [
    None, 256, 512, 1024, 2048, 4096, 8192, 16384
]:
    for cot in [True, False]:
        config = TrainConfig(
            name=f"{file_name}_ntokens={num_tokens}_cot{cot}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),
            dataset=CapsuleDataset.Config(
                data_sources=[
                    # (n=32768) with fixed logits
                    ("hazy-research/capsules/m03d17_generate_longhealth_p01:v0", None),
                ],  
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),
            generate_every_n_steps=16,
            generate_datasets=[
                GenerateDatasetConfig(
                    name_for_wandb="multiple_choice_generations",
                    dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                        patient_ids=["patient_01"],
                        max_questions=128,
                        cot=cot
                    )
                ),
            ],
            eval_every_n_steps=16,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb="longhealth_multiple_choice",
                    local_batch_size=16,
                    dataset=LongHealthEvalDataset.Config(
                        patient_ids=["patient_01"],
                        max_questions=256,
                        label_type="tokens",
                        data_sources=[]  # ignore this arg
                    )
                ),
                EvalDatasetConfig(
                    name_for_wandb="generated_questions",
                    local_batch_size=16,
                    dataset=CapsuleDataset.Config(
                        data_sources=[
                            ("hazy-research/capsules/m03d12_longhealth_p01_basic_qa_test:v0", None),
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
            epochs=0,
            lr=5e-3,
            wandb=WandBConfig(
                project="capsules",
                tags=["development", "baseline"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            train_batch_size=6,
        )
        configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
