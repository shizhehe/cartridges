import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.optim import CosWithWarmup
from cartridges.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.tasks.reglab import ReglabHousingQAGenerateDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset, CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]


if NUM_PATIENTS == 3:
    data_sources = [
        ("/data/sabri/code/Cartridges/output/2025-04-28-07-38-34-m04d28_generate_longhealth_generated_reformat_w_toc_px_no_instructions/86b02ca9-e4e5-4119-95b2-9b9c5f61d358/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-28-07-55-53-m04d28_generate_longhealth_generated_reformat_w_toc_px_no_instructions/19b0c793-656f-4ef9-941b-7dbf86e5dce7/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-28-08-13-54-m04d28_generate_longhealth_generated_reformat_w_toc_px_no_instructions/5dfbae53-e171-4f61-9e95-f4f68cc24bd2/artifact/dataset.pkl", None)
    ]

elif NUM_PATIENTS == 10:
    data_sources = [
        ("/data/sabri/code/Cartridges/output/2025-04-28-08-55-20-m04d28_generate_longhealth_generated_reformat_w_toc_px_no_instructions/eb70860d-9c31-4b7c-8c9e-5c245e62715d/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-28-09-26-16-m04d28_generate_longhealth_generated_reformat_w_toc_px_no_instructions/789e8ef5-e416-408c-b6a8-782f2d8ab816/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-28-09-59-31-m04d28_generate_longhealth_generated_reformat_w_toc_px_no_instructions/1e5d8199-1c0c-44e9-8795-37f71fd160f6/artifact/dataset.pkl", None),
    ]
else:
    raise ValueError(f"Invalid number of patients: {NUM_PATIENTS}")


configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_patients{patients_str}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        
        lr=lr,

        dataset=CartridgeDatasetLatest.Config(
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
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=patient_ids, 
                    cot=True,
                ),
                name_for_wandb=f"longhealth_mc",
                num_samples=8,
                temperature=0.3
            ),
        ],
        eval_every_n_steps=64,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb="longhealth_mc",
                local_batch_size=16,
                dataset=LongHealthEvalDataset.Config(
                    patient_ids=patient_ids,
                    max_questions=256,
                    label_type="tokens",
                    data_sources=[]  # ignore this arg
                )
            )
        ],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        epochs=1,

        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "longhealth", f"patients{patients_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=4,
    )
    for lr in [8e-3, 6e-3, 4e-3, 2e-3, 8e-4, 6e-4]
    for num_tokens in [2048] #2048, 4096, 6144]
]

if __name__ == "__main__":
    pydrantic.main(configs)
