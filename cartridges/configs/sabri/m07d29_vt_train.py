import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.ruler.evals import VariableTrackingGenerateDataset
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
        "/data/sabri/cartridges/2025-07-29-19-22-07-m07d29_vt_synthesize/m07d29_vt_synthesize_llama-3.2-3b_n65536-0/artifact/dataset.pkl"
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

# Use the variable tracking dataset path
BASE_PATH = "/home/sabri/code/cartridges/cartridges/data/ruler/_data"
vt_path = f"{BASE_PATH}/qwen3_4b-l100000-n1-c64-h2-noise-9df65ada.json"

config = TrainConfig(
    model=model,
    kv_cache_initializer=KVFromRandomText.Config(
        max_tokens=NUM_TOKENS
    ),
    
    lr=2e-2,
    loss_type="logits",
    epochs=2,
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
            num_samples=8,
            override_max_tokens=1024,
            temperature=0.3,
            batch_size=16,
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
    name=FormatStringVariable(f"{file_name}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
)


if __name__ == "__main__":
    pydrantic.main(config)