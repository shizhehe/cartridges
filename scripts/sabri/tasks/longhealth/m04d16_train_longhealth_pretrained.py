import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.kv_initialization.strategies.pretrained import KVCacheFromPretrained
from cartridges.optim import CosWithWarmup
from cartridges.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.tasks.reglab import ReglabHousingQAGenerateDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset, CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64

PATIENT_IDXS = [1, 2, 3]
patients_str = ''.join(f"p{idx:02d}" for idx in PATIENT_IDXS)  # used for names and tags
patient_ids = [f"patient_{idx:02d}" for idx in PATIENT_IDXS]

m04d16_data_sources = [
    ("hazy-research/Cartridges/m04d16_generate_longhealth_qa_p01p02p03_n32768:v1", None),
    ("hazy-research/Cartridges/m04d16_generate_longhealth_qa_p01p02p03_n32768:v2", None),
    ("hazy-research/Cartridges/m04d16_generate_longhealth_qa_p01p02p03_n32768:v3", None),
    ("hazy-research/Cartridges/m04d16_generate_longhealth_fair_section_p01p02p03_n32768:v1", 8192),    
]

m04d19_data_sources = [
    ("hazy-research/Cartridges/m04d16_generate_longhealth_qa_p01p02p03_n32768:v0", None),
    ("/data/sabri/code/Cartridges/output/2025-04-19-15-24-03-m04d19_generate_longhealth_reformat/b81d8fbc-da07-439d-a1a5-cdad74f6f2d6/artifact/dataset.pkl", None),
    ("/data/sabri/code/Cartridges/output/2025-04-19-15-37-42-m04d19_generate_longhealth_reformat/d9426f98-3ab4-47b4-ad2f-2548edf06967/artifact/dataset.pkl", None),
    (" /data/sabri/code/Cartridges/output/2025-04-19-15-46-30-m04d19_generate_longhealth_reformat/8f92f32f-06f6-4e65-8d7c-a5ccf29e4a67/artifact/dataset.pkl", None), # little bit of temperatureeee
],


m04d20_data_sources = [
    ("hazy-research/Cartridges/m04d16_generate_longhealth_qa_p01p02p03_n32768:v3", None),

    # mostly notes (0.8)
    # ("/data/sabri/code/Cartridges/output/2025-04-20-15-54-55-m04d20_generate_longhealth_generated_reformat_w_toc/0cc7d2f1-5eda-4250-8b68-2c78f15e8533/artifact/dataset.pkl", None),
    # ("/data/sabri/code/Cartridges/output/2025-04-20-16-02-04-m04d20_generate_longhealth_generated_reformat_w_toc/eeca2b77-e365-4585-b99d-9d6d5a72b590/artifact/dataset.pkl", None),
    # ("/data/sabri/code/Cartridges/output/2025-04-20-16-31-02-m04d20_generate_longhealth_generated_reformat_w_toc/de601b71-65e1-406c-8423-e26813e1601b/artifact/dataset.pkl", None),

    # mostly full records (0.1)
    ("/data/sabri/code/Cartridges/output/2025-04-20-17-02-02-m04d20_generate_longhealth_generated_reformat_w_toc/ba37d1b8-4701-4ed0-bc73-4a07de98050d/artifact/dataset.pkl", None),
    ("/data/sabri/code/Cartridges/output/2025-04-20-17-02-02-m04d20_generate_longhealth_generated_reformat_w_toc/ba37d1b8-4701-4ed0-bc73-4a07de98050d/artifact/dataset.pkl", None)
]


configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_patients{patients_str}_lr{{lr}}_id{wandb_run_id}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
        ),
        
        lr=5e-3,

        dataset=CartridgeDatasetLatest.Config(
            data_sources=m04d20_data_sources,
            max_sequence_length=1024,
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        
        save_every_n_steps=128,
        save_after_training=False,
        generate_every_n_steps=64,
        generate_max_new_tokens=512,
        generate_datasets=[
            GenerateDatasetConfig(
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=patient_ids, 
                    cot=True,
                ),
                name_for_wandb=f"longhealth_mc",
                num_samples=16,
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
        kv_cache_initializer=KVCacheFromPretrained.Config(
            wandb_run_id=f"hazy-research/Cartridges/{wandb_run_id}"
        ),
        loss_type="logits",
        epochs=0,

        wandb=WandBConfig(
            project="cartridges",
            tags=["eval", "longhealth", f"patients{patients_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=4,
    )
    for wandb_run_id in [
        # "pqxrilfq"  # 8192
        # "lgvzsyl4" # 2048
        # "n6qmmi6c" # 4096
        "btfay1cp" # 6144
    ]
]

if __name__ == "__main__":
    pydrantic.main(configs)
