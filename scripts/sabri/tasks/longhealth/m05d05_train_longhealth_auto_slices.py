import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.models.llama import LlamaForCausalLM
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

if "SLICES" in os.environ:
    SLICES = os.environ["SLICES"].split(",")
else:
    SLICES = [
        "structuring",
        "question",
    ]

if NUM_PATIENTS == 10:
    data_sources = {
        "structuring": [
            "/data/sabri/code/Cartridges/output/2025-05-05-09-05-55-m05d04_generate_longhealth_auto_slices/48a29aef-2623-4d16-999d-22a7eeb6fcdd/artifact/dataset.pkl",
            "/data/sabri/code/Cartridges/output/2025-05-05-09-24-06-m05d04_generate_longhealth_auto_slices/m05d04_generate_longhealth_auto_slices_p10_structuring65536-0/artifact/dataset.pkl",
        ],
        "question": [
            # Without the details in the question
            # "/data/sabri/code/Cartridges/output/2025-05-05-10-25-13-m05d04_generate_longhealth_auto_slices/m05d04_generate_longhealth_auto_slices_p10_question_65536-0/artifact/dataset.pkl",
            # "/data/sabri/code/Cartridges/output/2025-05-05-12-00-34-m05d04_generate_longhealth_auto_slices/m05d04_generate_longhealth_auto_slices_p10_question_n65536_cot0.5-0/artifact/dataset.pkl",
            # "/data/sabri/code/Cartridges/output/2025-05-05-12-18-05-m05d04_generate_longhealth_auto_slices/m05d04_generate_longhealth_auto_slices_p10_question_n65536_cot0.2-0/artifact/dataset.pkl",

            # With the details in the question
            "/data/sabri/code/Cartridges/output/2025-05-05-13-19-57-m05d04_generate_longhealth_auto_slices/m05d04_generate_longhealth_auto_slices_p10_question_n65536_cot0.2-0/artifact/dataset.pkl",
            "/data/sabri/code/Cartridges/output/2025-05-05-13-43-00-m05d04_generate_longhealth_auto_slices/m05d04_generate_longhealth_auto_slices_p10_question_n65536_cot0.5-0/artifact/dataset.pkl",
            
        ]
       
    }
elif NUM_PATIENTS == 20:
    data_sources = [
    ]
else:
    raise ValueError(f"Invalid number of patients: {NUM_PATIENTS}")




configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_{patients_str}_{'+'.join(SLICES)}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            model_cls=LlamaForCausalLM,
            attn_implementation="einsum",
        ),
        
        lr=lr,

        dataset=CartridgeDatasetLatest.Config(
            data_sources=[
                (source, None)
                for slc in SLICES
                for source in data_sources[slc]
            ],
            max_sequence_length=1024,
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        
        save_every_n_steps=512,
        generate_every_n_steps=512,
        generate_max_new_tokens=512,
        generate_datasets=[
            GenerateDatasetConfig(
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=patient_ids, 
                    cot=True,
                ),
                name_for_wandb=f"longhealth_mc",
                num_samples=4,
                num_samples_final=8,
                batch_size=16,
                temperature=0.3
            ),
        ],
        eval_every_n_steps=256,
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
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(max_tokens=num_tokens),
        loss_type="logits",
        epochs=2,

        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "longhealth", f"patients{patients_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        distributed_backend="gloo" if socket.gethostname().startswith("mk-ix-") else "nccl",
        global_batch_size=bs,
        local_batch_size=4,
    )
    for lr in [
        # 4e-2,
        2e-2,
        # , 1e-2, 9e-3, 8e-3
    ]
    # for lr in [8e-3]
    for num_tokens in [
        2048, 
        # 512,
        # 6144, 
        # 2048, 
        # 2048, 
        # 1024
    ]
]

if __name__ == "__main__":
    pydrantic.main(configs)
