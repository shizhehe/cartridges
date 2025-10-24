import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.models.llama import LlamaForCausalLM
from capsules.tasks.mmlu import MMLUEvalDataset, MMLUGenerateDataset
from capsules.tasks.qasper.context import QasperStructuredContextConfig
from capsules.tasks.qasper.dataset import QasperEvalDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig, PeftConfig
from capsules.datasets import CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


TOPIC = "question"


if TOPIC == "question":
    data_sources = [
        "hazy-research/capsules/generate_qasper_simple_question_s5_n65536:v1",
        "hazy-research/capsules/generate_qasper_simple_question_s5_n65536:v2",
    ]
else:
    raise ValueError(f"Invalid topic: {TOPIC}")

LORA_RANK = os.environ.get("LORA_RANK", "192")
LORA_RANK = list(map(int, LORA_RANK.split(",")))


configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_{TOPIC}_lr{{lr}}_rank{{model.peft.r}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            tuning_method="peft",
            peft=PeftConfig(
                enabled=True,  # Enable PEFT
                method="lora",  # Use LoRA
                r=rank,  # LoRA rank
                alpha=int(2 * rank),  # LoRA scaling factor (typically 2*rank)
                dropout=0.05,  # LoRA dropout
                # Updated target modules for LLaMA 3 architecture
                # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        ),
        
        lr=lr,

        dataset=CapsuleDatasetLatest.Config(
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
                    num_problems=1024,
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
        loss_type="logits",
        epochs=1,

        wandb=WandBConfig(
            project="capsules",
            tags=["train", "qasper", f"qasper-{TOPIC}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        distributed_backend="gloo",
        global_batch_size=bs,
        local_batch_size=4,
        use_batch_sampler=True,
        log_logprob_viz=False,

    )
    for lr in [
        # 4e-2,
        2e-4,
        # , 1e-2, 9e-3, 8e-3
    ]
    # for lr in [8e-3]
    for rank in LORA_RANK
]

if __name__ == "__main__":
    pydrantic.main(configs)
