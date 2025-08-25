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


NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "2048"))

LEVEL = 1

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "qwen":
    data_sources = [
        # "/data/sabri/cartridges/2025-08-15-18-19-24-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
        # "/data/sabri/cartridges/2025-08-16-11-06-58-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
        # "/data/sabri/cartridges/2025-08-16-13-24-57-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
        "/data/sabri/cartridges/2025-08-18-10-03-08-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"

    ]
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
    kv_cache_initializer=KVFromRandomText.Config(max_tokens=NUM_TOKENS)
elif MODEL == "llama":
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
    if LEVEL == 1:
        data_sources = [
            # "/data/sabri/cartridges/2025-08-18-10-25-20-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
            # "/data/sabri/cartridges/2025-08-18-10-50-58-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
            # "/data/sabri/cartridges/2025-08-18-10-58-55-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
            # "/data/sabri/cartridges/2025-08-19-16-31-35-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
            # "/data/sabri/cartridges/2025-08-20-10-33-32-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
            # "/data/sabri/cartridges/2025-08-20-17-24-04-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
            
            # round 1
            DataSource(
                path="/data/sabri/cartridges/2025-08-24-15-39-39-codehop_synthesize/codehop_synthesize_n8192-0/artifact/dataset.parquet",
                limit=8192,
                type="local"
            )
        ]
        kv_cache_initializer=KVFromRandomText.Config(max_tokens=NUM_TOKENS)

    elif ROUND == 2:
        data_sources = [
            # round 2
            "/data/sabri/cartridges/2025-08-22-15-59-22-codehop_synthesize/codehop_synthesize_n8192-0/artifact/dataset.parquet",
            "/data/sabri/cartridges/2025-08-22-16-14-40-codehop_synthesize/codehop_synthesize_n8192-0/artifact/dataset.parquet",
            "/data/sabri/cartridges/2025-08-23-12-10-38-codehop_synthesize/codehop_synthesize_n65768-0/artifact/dataset.parquet"
        ]
        # kv_cache_initializer = KVFromPretrained.Config(
        #     wandb_run_id="hazy-research/cartridges/85axrvk4",
        # )
        kv_cache_initializer=KVFromRandomText.Config(max_tokens=NUM_TOKENS)

            
else:
    raise ValueError(f"Invalid model: {MODEL}")


config = TrainConfig(
    model=model,
    kv_cache_initializer=kv_cache_initializer,
    
    lr=2e-2,
    epochs=1,
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
            dataset=CodeHopGenerateDataset.Config(make_run_dir=str(dataset_dir)),
            name_for_wandb=f"codehop",
            generate_max_new_tokens=8,
            batch_size=32,
            temperature=0.0,
        )
    ],
    distributed_backend="gloo",

    wandb=WandBConfig(tags=["train", "codehop"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(f"codehop_train_lr{{lr}}_toks{NUM_TOKENS}"),
)


if __name__ == "__main__":
    pydrantic.main(config)