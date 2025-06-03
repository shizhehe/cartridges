import os
from pathlib import Path

import pydrantic

from cartridges.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset
from cartridges.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
# for num_tokens in [128, 256, 512, 1024, 2048, 4096, 8192, 16384]:

for lr in [
    1e-5, 3e-5, 5e-6
    # 5e-3, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6
]:
    for pidx in range(5, 6):
        patient_id = f"patient_{pidx:02d}"
        config = TrainConfig(
            name=f"{file_name}_p{pidx:02d}_lr{lr:.0e}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),
            dataset=CartridgeDataset.Config(
                data_sources=[
                    (f"hazy-research/Cartridges/m03d19_generate_longhealth_sweep_patients_p{pidx:02d}:v0", None),
                ],  
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),
            generate_every_n_steps=64,
            generate_datasets=[
                GenerateDatasetConfig(
                    name_for_wandb="multiple_choice_generations",
                    dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                        patient_ids=[patient_id],
                        max_questions=128,
                        cot=True
                    )
                ),
            ],
            eval_every_n_steps=64,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb="longhealth_multiple_choice",
                    local_batch_size=16,
                    dataset=LongHealthEvalDataset.Config(
                        patient_ids=[patient_id],
                        max_questions=256,
                        label_type="tokens",
                        data_sources=[]  # ignore this arg
                    )
                ),
                # EvalDatasetConfig(
                #     name_for_wandb="generated_questions",
                #     batch_size=16,
                #     dataset=CartridgeDataset.Config(
                #         data_sources=[
                #             ("hazy-research/Cartridges/m03d12_longhealth_p01_basic_qa_test:v0", None),
                #         ],
                #         is_wandb=True,
                #         label_type="tokens",
                #     ),
                # )
            ],
            generate_max_new_tokens=1024,
            kv_cache_initalizer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=None,  # use full context
                num_frozen_tokens=1,
            ),
            loss_type="logits",
            save_every_n_steps=64,
            epochs=1,
            lr=lr,
            wandb=WandBConfig(
                project="cartridges",
                tags=["cache_tuning", "development"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
            train_batch_size=2,
        )
        configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
