import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.optim import CosWithWarmup
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.reglab import ReglabHousingQAGenerateDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]


if NUM_PATIENTS == 10:
    data_sources = [
        ("/data/sabri/code/capsules/output/2025-05-01-13-19-08-m05d01_generate_longhealth_auto/38dd60df-e262-4fde-a2e2-197756461a71/artifact/dataset.pkl", None),
        ("/data/sabri/code/capsules/output/2025-05-01-13-58-59-m05d01_generate_longhealth_auto/71dbbcf6-9788-4fa0-9fea-ae35ad947663/artifact/dataset.pkl", None)
    ]
elif NUM_PATIENTS == 20:
    data_sources = [
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

        dataset=CapsuleDatasetLatest.Config(
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
                num_samples=1,
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
            project="capsules",
            tags=["train", "longhealth", f"patients{patients_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=1,
    )
    for lr in [8e-3, 6e-3, 4e-3, 2e-3, 8e-4, 6e-4]
    for num_tokens in [2048] #2048, 4096, 6144]
]

if __name__ == "__main__":
    pydrantic.main(configs)
