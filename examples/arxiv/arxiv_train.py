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

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "2048"))

DATA_SOURCE = "/data/sabri/cartridges/2025-08-08-17-26-20-arxiv_synthesize/arxiv_synthesize_Qwen/Qwen3-4b_n8192-0/artifact/dataset.pkl"
model=HFModelConfig(
    pretrained_model_name_or_path="Qwen/Qwen3-4b",
    model_cls=FlexQwen3ForCausalLM,
)

config = TrainConfig(
    model=model,
    kv_cache_initializer=KVFromRandomText.Config(
        max_tokens=NUM_TOKENS
    ),
    
    lr=2e-2,
    epochs=2,
    global_batch_size=32,

    dataset=CartridgeTrainDataset.Config(
        data_sources=[(DATA_SOURCE, None)],
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
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