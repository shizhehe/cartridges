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
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]


NUM_TOKENS = os.environ.get("NUM_TOKENS", "2048")
NUM_TOKENS = list(map(int, NUM_TOKENS.split(",")))


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
        name=FormatStringVariable(f"{file_name}_{patients_str}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            model_cls=LlamaForCausalLM,
            attn_implementation="einsum",
        ),
        
        lr=lr,

        dataset=CartridgeDatasetLatest.Config(
            data_sources=[
                (source, None)
                for source in data_sources
            ],
            max_sequence_length=1024,
            is_wandb=True,
            label_type="tokens",
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
        loss_type="tokens",
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
        use_batch_sampler=True
    )
    for lr in [
        # 4e-2,
        # 3e-2,
        2e-2,
        # 1e-2
        # , 1e-2, 9e-3, 8e-3
    ]
    # for lr in [8e-3]
    for num_tokens in NUM_TOKENS
]

if __name__ == "__main__":
    pydrantic.main(configs)
