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
from cartridges.data.codehop.evals import CodeHopGenerateDataset
from cartridges.utils.wandb import WandBConfig

from cartridges.configs.codehop_synthesize import SYSTEM_PROMPT_TEMPLATE

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "4096"))
LEVEL = int(os.environ.get("LEVEL", "1"))
MODEL = os.environ.get("MODEL", "qwen")
REPO = os.environ.get("REPO", "244c02")

ENHANCING_DATASET_DIR = "/data/sabri/cartridges/2025-08-30-13-33-13-make_codehop/codehop-nf768-nm1-dc0-v5-fn36-0/repo-e30278"

REPOS = {
    "244c02": {
        "dataset_dir": "/data/sabri/cartridges/2025-08-26-16-23-39-make_codehop/codehop-nf16-nm1-dc3-v5-fn36-0/repo-244c02",
        "datasources": {
            "qwen": {
                1: [
                    # DataSource(path="codehop_synthesize_repo-244c02_level1_n65536:v0", type="wandb"),
                    # DataSource(path="codehop_synthesize_qwen_repo-244c02_level1_n1024:v0", type="wandb"),
                    DataSource(path="codehop_synthesize_qwen_repo-244c02_level1_n65768:v1", type="wandb"),
                ]
            },
            "llama": {
                1: [
                    DataSource(path="codehop_synthesize_llama_repo-244c02_level1_n65768:v4", type="wandb"),
                ],
                2: [
                    DataSource(path="codehop_synthesize_llama_repo-244c02_level2_n65768:v1", type="wandb"),
                ]
            }
        },
        "initializer": {
            "qwen": {
                1: KVFromRandomText.Config(max_tokens=NUM_TOKENS),
            },
            "llama": {
                1: KVFromRandomText.Config(max_tokens=NUM_TOKENS),
                2: KVFromPretrained.Config(wandb_run_id="hazy-research/cartridges/1mreremx"),
            },
        }
    }
}
dataset_dir = Path(REPOS[REPO]["dataset_dir"]).parent
filename = Path(__file__).stem


if MODEL == "qwen":

    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
elif MODEL == "llama":
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")


config = TrainConfig(
    model=model,
    kv_cache_initializer=REPOS[REPO]["initializer"][MODEL][LEVEL],
    
    lr=2e-2,
    epochs=2,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=REPOS[REPO]["datasources"][MODEL][LEVEL],
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
        ), 
        GenerationEvalConfig(
            dataset=CodeHopGenerateDataset.Config(
                make_run_dir=str(dataset_dir), 
                enhancing_dir=str(ENHANCING_DATASET_DIR),
                enhancing_system_prompt_template=SYSTEM_PROMPT_TEMPLATE,
            ),
            name_for_wandb=f"codehop_w_ctx",
            generate_max_new_tokens=64,
            batch_size=16,
            temperature=0.0,
        )
    ],
    distributed_backend="gloo",

    wandb=WandBConfig(tags=["train", "codehop"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=FormatStringVariable(f"{filename}_{MODEL}_{REPO}_level{LEVEL}_lr{{lr}}_toks{NUM_TOKENS}"),
)


if __name__ == "__main__":
    pydrantic.main(config)