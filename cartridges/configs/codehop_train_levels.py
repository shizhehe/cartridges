import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization import KVFromRandomText, KVFromPretrained
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig, DataSource
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset
from cartridges.data.codehopv2.evals import CodeHopGenerateDataset
from cartridges.utils.wandb import WandBConfig

from cartridges.configs.codehop_synthesize import DATASET_DIR
dataset_dir = Path(DATASET_DIR).parent
filename = Path(__file__).stem


NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "2048"))

LEVEL = 1

MODEL = os.environ.get("MODEL", "qwen")
if MODEL == "qwen":

    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
    if LEVEL == 1:
        data_sources = [
            # "/data/sabri/cartridges/2025-08-24-15-39-39-codehop_synthesize/codehop_synthesize_n8192-0/artifact/dataset.parquet"
            DataSource(path="codehop_synthesize_n65768:v1", type="wandb")
        ]
    elif LEVEL == 2:
        raise NotImplementedError("Level 2 not implemented")
    kv_cache_initializer=KVFromRandomText.Config(max_tokens=NUM_TOKENS)
elif MODEL == "llama":
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
    if LEVEL == 1:
        kv_cache_initializer=KVFromRandomText.Config(max_tokens=NUM_TOKENS)
else:
    raise ValueError(f"Invalid model: {MODEL}")


config = TrainConfig(
    model=model,
    kv_cache_initializer=kv_cache_initializer,
    
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
    generate_eval_every_n_steps=16,
    generate_evals=[
        GenerationEvalConfig(
            dataset=CodeHopGenerateDataset.Config(make_run_dir=str(dataset_dir)),
            name_for_wandb=f"codehop",
            generate_max_new_tokens=64,
            batch_size=32,
            temperature=0.0,
        )
    ],
    distributed_backend="gloo",

    wandb=WandBConfig(tags=["train", "codehop"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(f"{filename}_lr{{lr}}_toks{NUM_TOKENS}"),
)


if __name__ == "__main__":
    pydrantic.main(config)