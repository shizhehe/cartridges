import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization.pretrained import KVCacheFromPretrained
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

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))
MODEL_SIZE = os.environ.get("MODEL_SIZE", "8")
size_str = MODEL_SIZE.replace(".", "_")

MODEL = "qwen"

model=HFModelConfig(
    pretrained_model_name_or_path=f"Qwen/Qwen3-{MODEL_SIZE}b",
    model_cls=FlexQwen3ForCausalLM,
)

if MODEL_SIZE == "0.6":
    data_sources = [
        "/data/sabri/cartridges/2025-07-29-13-00-47-m07d29_longhealth_synthesize_sizes/m07d29_longhealth_synthesize_sizes_qwen3-0.6b_p10_n131072-0/artifact/dataset.pkl"
    ]
elif MODEL_SIZE == "1.7":
    data_sources = [
        "/data/sabri/cartridges/2025-07-29-13-00-47-m07d29_longhealth_synthesize_sizes/m07d29_longhealth_synthesize_sizes_qwen3-1.7b_p10_n131072-1/artifact/dataset.pkl"
    ]
elif MODEL_SIZE == "4":
    data_sources = [
        "/data/sabri/cartridges/2025-07-27-14-11-52-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0/artifact/dataset.pkl",
        "/data/sabri/cartridges/2025-07-27-15-00-07-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0/artifact/dataset.pkl"
    ]
elif MODEL_SIZE == "8":
    data_sources = [
        # "/data/sabri/cartridges/2025-07-29-13-00-47-m07d29_longhealth_synthesize_sizes/m07d29_longhealth_synthesize_sizes_qwen3-8b_p10_n131072-2/artifact/dataset.pkl"
        "/data/sabri/cartridges/2025-07-30-09-21-14-m07d29_longhealth_synthesize_sizes/m07d29_longhealth_synthesize_sizes_qwen3-8b_p10_n131072-0/artifact/dataset.pkl"
    ]
else:
    raise ValueError(f"Invalid model: {MODEL_SIZE}")


config = TrainConfig(
    model=model,
    kv_cache_initializer=KVCacheFromPretrained.Config(
        wandb_run_id="hazy-research/cartridges/cqx2uc01",
    ),
    
    lr=2e-2,
    loss_type="logits",
    epochs=6,
    global_batch_size=32,

    dataset=CartridgeTrainDataset.Config(
        data_sources=[(source, None) for source in data_sources],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_every_n_steps=512,
    generate_evals=[
        GenerationEvalConfig(
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids,
            ),
            name_for_wandb=f"longhealth_{patients_str}",
            generate_max_new_tokens=2048,
            batch_size=32,
            temperature=0.3,
        )
    ],
    eval_every_n_steps=512,
    eval_datasets=[],
    distributed_backend="gloo",

    wandb=WandBConfig(
        project="cartridges",
        tags=["train", "longhealth"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"{file_name}_lr{{lr}}_size{size_str}"),
)


if __name__ == "__main__":
    pydrantic.main(config)