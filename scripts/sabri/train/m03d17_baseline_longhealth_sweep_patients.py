import os
from pathlib import Path

import pydrantic

from cartridges.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset
from cartridges.tasks.longhealth import LongHealthContextConfig, LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
for patient_id in [f"patient_{idx:02d}" for idx in range(9, 21)]:
    for num_tokens in [
        None,
        256, 512, 1024, 2048, 4096, 8192, 16384
    ]:
        config = TrainConfig(
            name=f"{file_name}_ntokens-{num_tokens}_patient-{patient_id}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),
            dataset=CartridgeDataset.Config(
                data_sources=[
                    # (n=32768) with fixed logits
                    ("hazy-research/Cartridges/m03d17_generate_longhealth_p01:v0", None),
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
                        patient_ids=[patient_id],
                        max_questions=128,
                        cot=True,
                        include_diagnosis=True
                    )
                ),
            ],
            eval_every_n_steps=16,
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
            ],
            generate_max_new_tokens=1024,
            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=num_tokens,
                context=LongHealthContextConfig(
                    patient_ids=[patient_id]
                )
            ),
            loss_type="logits",
            save_every_n_steps=64,
            epochs=0,
            lr=5e-3,
            wandb=WandBConfig(
                project="cartridges",
                tags=["development", "baseline"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
            train_batch_size=6,
        )
        configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)
