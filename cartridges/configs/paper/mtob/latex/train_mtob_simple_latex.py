import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.models.llama import LlamaForCausalLM
from capsules.optim import CosWithWarmup
from capsules.tasks.longhealth import (
    LongHealthEvalDataset,
    LongHealthMultipleChoiceGenerateDataset,
)
from capsules.tasks.longhealth.context import LongHealthStructuredContextConfig
from capsules.tasks.mmlu import MMLUGenerateDataset
from capsules.tasks.mtob import mtob_generate_datasets
from capsules.tasks.mtob.context import MTOBNoStructuredContext
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64

NUM_TOKENS = os.environ.get("NUM_TOKENS", "2048")
NUM_TOKENS = list(map(int, NUM_TOKENS.split(",")))

LR = os.environ.get("LR", "1e-2")
LR = list(map(float, LR.split(",")))

# TOKENS_TO_LR = {8192: 0.001, 4096: 0.005, 2048: 0.005, 1024: 0.001, 512: 0.05, 256: 0.05, 128: 0.05}


configs = [
    TrainConfig(
        name=FormatStringVariable(
            f"{file_name}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"
        ),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
            model_cls=LlamaForCausalLM,
            attn_implementation="einsum",
        ),
        lr=lr,
        dataset=CapsuleDatasetLatest.Config(
            data_sources=[
                ("hazy-research/capsules/generate_mtob_simple_latex_s5_n65536:v0", None),
                ("hazy-research/capsules/generate_mtob_simple_latex_s5_n65536:v1", None),
            ],
            max_sequence_length=1024,
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        context=MTOBNoStructuredContext(setup="latex_and_sentences"),
        save_every_n_steps=512,
        generate_every_n_steps=64,
        generate_max_new_tokens=64,
        generate_datasets=[
            *mtob_generate_datasets(),
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
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        epochs=1,
        wandb=WandBConfig(
            project="capsules",
            tags=["train", "mtob"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        distributed_backend=(
            "gloo" 
        ),
        global_batch_size=bs,
        local_batch_size=2,
        use_batch_sampler=True,
    )
    for num_tokens in NUM_TOKENS
    for lr in LR
]

if __name__ == "__main__":
    pydrantic.main(configs)
