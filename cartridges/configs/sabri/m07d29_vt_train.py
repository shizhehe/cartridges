import os
from pathlib import Path
import socket
from typing import Optional

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.ruler.evals import VariableTrackingGenerateDataset
from cartridges.initialization.pretrained import KVCacheFromPretrained
from cartridges.initialization.random import KVFromRandomText, KVFromRandomVectors
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeTrainDataset
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem
bs = 4

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "llama":
    # Placeholder data sources - will need to be updated with actual VT synthesis datasets
    DATA_SOURCES = [
        # "/data/sabri/cartridges/2025-07-29-14-51-14-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl"
        # "/data/sabri/cartridges/2025-07-29-18-47-55-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl"
        # "/data/sabri/cartridges/2025-07-29-19-22-07-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl"
        # "/data/sabri/cartridges/2025-08-04-10-38-54-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl"
        # "/data/sabri/cartridges/2025-08-04-12-44-24-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl"
        "/data/sabri/cartridges/2025-08-04-14-36-31-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",
        "/data/sabri/cartridges/2025-08-04-15-00-22-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl",
        "/data/sabri/cartridges/2025-08-04-15-53-44-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl"
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
elif MODEL == "qwen":
    DATA_SOURCES = [
        "/data/sabri/cartridges/2025-08-04-15-10-52-m07d29_vt_synthesize/m07d29_vt_synthesize_qwen3-4b_n65536-0/artifact/dataset.pkl",
        "/data/sabri/cartridges/2025-08-04-15-24-14-m07d29_vt_synthesize/m07d29_vt_synthesize_qwen3-4b_n65536-0/artifact/dataset.pkl"
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4B",
        model_cls=FlexQwen3ForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

# Use the variable tracking dataset path
BASE_PATH = "/home/sabri/code/cartridges/cartridges/data/ruler/_data"
# vt_path = f"{BASE_PATH}/qwen3_4b-l100000-n1-c64-h2-noise-9df65ada.json"
# vt_path = f"{BASE_PATH}/llama_3.2_3b_instruct-l100000-n1-c64-h2-essay-7ba69bcb.json"
vt_path = f"{BASE_PATH}/llama_3.2_3b_instruct-l10000-n1-c16-h2-essay-words-1d31e1f5.json"

PRETRAINED_WANDB_RUN_ID: Optional[str] = os.environ.get("PRETRAINED_WANDB_RUN_ID", None)

if PRETRAINED_WANDB_RUN_ID:
    kv_cache_initializer = KVCacheFromPretrained.Config(
        wandb_run_id=PRETRAINED_WANDB_RUN_ID,
    )
else:
    kv_cache_initializer = KVFromRandomText.Config(
        max_tokens=NUM_TOKENS
    )

config = TrainConfig(
    model=model,
    kv_cache_initializer=kv_cache_initializer,
    lr=2e-2,
    epochs=5,
    global_batch_size=32,

    dataset=CartridgeTrainDataset.Config(
        data_sources=[
            (source, None)
            for source in DATA_SOURCES
        ],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=VariableTrackingGenerateDataset.Config(
                variable_tracking_path=vt_path,
                thinking=True,
            ),
            name_for_wandb=f"variable_tracking",
            num_samples=1,
            override_max_tokens=1024,
            temperature=0.0,
            batch_size=32,
        ),
    ],
    eval_every_n_steps=512,
    eval_datasets=[],
    distributed_backend="gloo",

    wandb=WandBConfig(
        project="cartridges",
        tags=["train", "variable_tracking"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"{file_name}_lr{{lr}}"),
)


if __name__ == "__main__":
    pydrantic.main(config)