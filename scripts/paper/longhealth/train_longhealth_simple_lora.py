import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.models.llama import LlamaForCausalLM
from cartridges.optim import CosWithWarmup
from cartridges.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig
from cartridges.tasks.mmlu import MMLUEvalDataset, MMLUGenerateDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig, PeftConfig
from cartridges.datasets import CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]


LORA_RANK = os.environ.get("LORA_RANK", "192")
LORA_RANK = list(map(int, LORA_RANK.split(",")))


if NUM_PATIENTS == 10:
    data_sources = [
        "hazy-research/Cartridges/generate_longhealth_simple_p10_s5_n65536:v0",
        "hazy-research/Cartridges/generate_longhealth_simple_p10_s5_n65536:v1",
        
        # with desc and 4096 max chunk size
        # "hazy-research/Cartridges/generate_longhealth_simple_p10_s5_n65536:v3",
        # "hazy-research/Cartridges/generate_longhealth_simple_p10_s5_n65536:v4"
    ]
elif NUM_PATIENTS == 20:
    data_sources = [
    ]
else:
    raise ValueError(f"Invalid number of patients: {NUM_PATIENTS}")



configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_{patients_str}_lr{{lr}}_rank{{model.peft.r}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            tuning_method="peft",
            peft=PeftConfig(
                enabled=True,  # Enable PEFT
                method="lora",  # Use LoRA
                r=rank,  # LoRA rank
                alpha=int(2 * rank),  # LoRA scaling factor (typically 2*rank)
                dropout=0.05,  # LoRA dropout
                # Updated target modules for LLaMA 3 architecture
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        ),
        
        lr=lr,

        dataset=CartridgeDatasetLatest.Config(
            data_sources=[
                (source, None)
                for source in data_sources
            ],
            max_sequence_length=1024,
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),

        context=LongHealthStructuredContextConfig(patient_ids=patient_ids),
        
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
                num_samples=8,
                num_samples_final=8,
                batch_size=16,
                temperature=0.3
            ),
            GenerateDatasetConfig(
                dataset=MMLUGenerateDataset.Config(
                    num_problems=512,
                ),
                batch_size=64,
                name_for_wandb=f"mmlu"
            )
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
            ),
            EvalDatasetConfig(
                name_for_wandb="mmlu",
                local_batch_size=16,
                dataset=MMLUEvalDataset.Config(num_samples=512),
            ),
        ],
        loss_type="logits",
        epochs=2,

        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "longhealth", f"patients{patients_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        distributed_backend="gloo", # if socket.gethostname().startswith("mk-ix-") else "nccl",
        global_batch_size=bs,
        local_batch_size=4,
        use_batch_sampler=True
    )
    for lr in [
        # 4e-2,
        # 3e-2,
        # 1e-2,
        # 2e-3,
        
        # 6e-4,
        # 5e-4,
        # 4e-4,
        # 3e-4,
        2e-4,
        # 1e-4, 
        # 9e-5,
        # 8e-5,
        # 7e-5
        # 2e-5,
        # 1e-2
        # , 1e-2, 9e-3, 8e-3
    ]
    # for lr in [8e-3]
    for rank in LORA_RANK
]
# 204, 408, 816, 1632, 3264, 6528, 13056, 26112, 52224, 104448

if __name__ == "__main__":
    pydrantic.main(configs)
