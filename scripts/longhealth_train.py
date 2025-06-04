import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.models.llama import LlamaForCausalLM
from cartridges.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig
from cartridges.tasks.mmlu import MMLUGenerateDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeTrainDataset
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem
bs = 64

NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

data_sources = [
    "/data/sabri/cartridges/2025-06-03-13-37-17-longhealth_synthesize/longhealth_synthesize_p10_n65536-0/artifact/dataset.pkl"
]

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=LlamaForCausalLM,
        attn_implementation="einsum",
    ),
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(max_tokens=2048),
    
    lr=2e-2,
    loss_type="logits",
    epochs=2,
    global_batch_size=bs,
    local_batch_size=4,
    use_batch_sampler=True,

    dataset=CartridgeTrainDataset.Config(
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
    eval_datasets=[],
    distributed_backend="gloo",

    wandb=WandBConfig(
        project="cartridges",
        tags=["train", "longhealth", f"patients{patients_str}"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"{file_name}_{patients_str}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
)


if __name__ == "__main__":
    pydrantic.main(config)
