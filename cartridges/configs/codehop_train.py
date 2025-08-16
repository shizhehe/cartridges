import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization import KVFromRandomText
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset
from cartridges.data.codehop.evals import CodeHopGenerateDataset
from cartridges.utils.wandb_utils import WandBConfig

from cartridges.configs.codehop_synthesize import DATASET_DIR


NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "2048"))

MODEL = os.environ.get("MODEL", "qwen")
if MODEL == "qwen":
    data_sources = [
        "/data/sabri/cartridges/2025-08-15-18-06-06-codehop_synthesize/codehop_synthesize_n8192-0/artifact/dataset.parquet",
        "/data/sabri/cartridges/2025-08-15-18-19-24-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
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
        max_tokens=NUM_TOKENS
    ),
    
    lr=2e-2,
    epochs=2,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=data_sources,
        top_k_logits=20,
        packed_seq_length=2048,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_eval_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=CodeHopGenerateDataset.Config(make_run_dir=DATASET_DIR),
            name_for_wandb=f"codehop",
            generate_max_new_tokens=8,
            batch_size=32,
            temperature=0.0,
        )
    ],
    distributed_backend="gloo",

    wandb=WandBConfig(tags=["train", "codehop"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable("codehop_train_lr{lr}_toks{kv_cache_initializer.max_tokens}"),
)


if __name__ == "__main__":
    pydrantic.main(config)