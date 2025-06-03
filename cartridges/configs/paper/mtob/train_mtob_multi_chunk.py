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
from capsules.tasks.mtob import mtob_generate_datasets
from capsules.tasks.mtob.context import MTOBNoStructuredContext
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64



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
            data_sources=[('hazy-research/capsules/generate_mtob_multi_chunk_s5_n65536:v0', None)],
            max_sequence_length=1024,
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        context=MTOBNoStructuredContext(setup="medium_and_sentences"),
        save_every_n_steps=512,
        generate_every_n_steps=64,
        generate_max_new_tokens=64,
        generate_datasets=[
            *mtob_generate_datasets(),
        ],
        eval_every_n_steps=256,
        eval_datasets=[],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        epochs=2,
        wandb=WandBConfig(
            project="capsules",
            tags=["train", "mtob"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        distributed_backend=(
            "gloo" if socket.gethostname().startswith("mk-ix-") else "nccl"
        ),
        global_batch_size=bs,
        local_batch_size=1,
    )
    for lr in [
        # 4e-2,
        2e-2,
        # , 1e-2, 9e-3, 8e-3
    ]
    # for lr in [8e-3]
    for num_tokens in [
        2048,
        # 512,
        # 6144,
        # 2048,
        # 2048,
        # 1024
    ]
]

if __name__ == "__main__":
    pydrantic.main(configs)
