import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.models.llama import LlamaForCausalLM
from cartridges.tasks.mmlu import MMLUEvalDataset, MMLUGenerateDataset
from cartridges.tasks.qasper.context import QasperStructuredContextConfig
from cartridges.tasks.qasper.dataset import QasperEvalDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


TOPIC = "question"


if TOPIC == "question":
    data_sources = [
        "hazy-research/Cartridges/generate_qasper_simple_no_slices_question_s1_n65536:v0",
        "hazy-research/Cartridges/generate_qasper_simple_no_slices_question_s1_n65536:v1",
    ]
else:
    raise ValueError(f"Invalid topic: {TOPIC}")


NUM_TOKENS = os.environ.get("NUM_TOKENS", "2048")
NUM_TOKENS = list(map(int, NUM_TOKENS.split(",")))


configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_{TOPIC}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            model_cls=LlamaForCausalLM,
            attn_implementation="einsum",
        ),
        
        lr=lr,

        dataset=CartridgeDatasetLatest.Config(
            data_sources=[
                (source, None)
                for source in data_sources
            ],
            max_sequence_length=1024,
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),

        context=QasperStructuredContextConfig(topic=TOPIC),
        
        save_every_n_steps=512,
        generate_every_n_steps=512,
        generate_max_new_tokens=512,
        generate_datasets=[
            GenerateDatasetConfig(
                dataset=MMLUGenerateDataset.Config(
                    num_problems=512,
                ),
                batch_size=64,
                name_for_wandb=f"mmlu"
            )
        ],
        eval_every_n_steps=256,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb="qasper_rewrite",
                local_batch_size=16,
                dataset=QasperEvalDataset.Config(
                    dataset="sabrieyuboglu/qasper-rewrite-gpt-4.1",
                    topic=TOPIC,
                    label_type="tokens",
                    data_sources=[]  # ignore this arg
                )
            ),
            EvalDatasetConfig(
                name_for_wandb="mmlu",
                local_batch_size=16,
                dataset=MMLUEvalDataset.Config(num_samples=512),
            ),
        ],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(max_tokens=num_tokens),
        loss_type="logits",
        epochs=1,

        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "qasper", f"qasper-{TOPIC}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        distributed_backend="gloo",
        global_batch_size=bs,
        local_batch_size=4,
        use_batch_sampler=True,
        log_logprob_viz=False,

    )
    for lr in [
        # 4e-2,
        2e-2,
        # , 1e-2, 9e-3, 8e-3
    ]
    # for lr in [8e-3]
    for num_tokens in NUM_TOKENS
]

if __name__ == "__main__":
    pydrantic.main(configs)
