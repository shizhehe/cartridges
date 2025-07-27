import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization.random import KVFromRandomText, KVFromRandomVectors
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeTrainDataset
from cartridges.utils import WandBConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset

file_name = Path(__file__).stem
bs = 4

NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]



MODEL = os.environ.get("MODEL", "qwen")
if MODEL == "llama":
    data_sources = [
        "/data/sabri/cartridges/2025-07-26-12-21-32-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0/artifact/dataset.pkl",
        "/data/sabri/cartridges/2025-07-26-13-40-44-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0/artifact/dataset.pkl"
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
elif MODEL == "qwen":
    data_sources = [
        "/data/sabri/cartridges/2025-07-26-12-02-19-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0/artifact/dataset.pkl"
    ]
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

config = TrainConfig(
    model=model,
    kv_cache_initializer=KVFromRandomText.Config(
        max_tokens=8192
    ),
    
    lr=2e-2,
    loss_type="logits",
    epochs=2,
    global_batch_size=32,

    dataset=CartridgeTrainDataset.Config(
        data_sources=[
            (source, None)
            for source in data_sources
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids,
            ),
            name_for_wandb=f"longhealth_{patients_str}",
            generate_max_new_tokens=1024,
            batch_size=32,
            temperature=0.3,
        )
    ],
    eval_every_n_steps=256,
    eval_datasets=[],
    distributed_backend="gloo",

    wandb=WandBConfig(
        project="cartridges",
        tags=["train", "longhealth"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"{file_name}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
)


if __name__ == "__main__":
    pydrantic.main(config)