import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
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
OFFSET = 10
# patient_idxs = list(range(OFFSET + 1, NUM_PATIENTS + OFFSET))
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]


if NUM_PATIENTS == 10:
    data_sources = [
        ("/data/sabri/code/Cartridges/output/2025-04-22-15-10-19-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/58acef64-5991-4174-8f6c-25de7a817596/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-22-19-28-13-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/75cec0ba-ab2b-4542-a114-99fb679b44eb/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-22-20-40-40-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/cc2c97e1-eb6a-467a-b86b-b541c148fed0/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-22-21-32-05-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/559b979b-c2b4-44dd-98b8-6be995510561/artifact/dataset.pkl", None)
    ]
elif NUM_PATIENTS == 20:
    data_sources = [
        ("/data/sabri/code/Cartridges/output/2025-04-23-12-39-21-m04d22_generate_longhealth_generated_reformat_w_toc_px/0409e626-eb99-4d98-972a-a369cfc42bd4/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-23-13-49-18-m04d22_generate_longhealth_generated_reformat_w_toc_px/1afd9c12-4980-4ea0-9cc5-cf293509fe89/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-23-16-13-22-m04d22_generate_longhealth_generated_reformat_w_toc_px/1f548a61-0256-4bb7-87ba-135405a8b82d/artifact/dataset.pkl", None),
        ("/data/sabri/code/Cartridges/output/2025-04-23-17-37-58-m04d22_generate_longhealth_generated_reformat_w_toc_px/e0869e24-3caa-41b9-8b26-8e87b0f21a71/artifact/dataset.pkl", None)
    ]
else:
    raise ValueError(f"Invalid number of patients: {NUM_PATIENTS}")


configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_opp_patients{patients_str}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
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
