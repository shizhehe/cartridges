import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.models.llama import LlamaForCausalLM
from cartridges.optim import CosWithWarmup
from cartridges.tasks.longhealth import (
    LongHealthEvalDataset,
    LongHealthMultipleChoiceGenerateDataset,
)
from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig
from cartridges.tasks.mmlu import MMLUGenerateDataset
from cartridges.tasks.mtob import mtob_generate_datasets
from cartridges.tasks.mtob.context import MTOBNoStructuredContext
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig, PeftConfig
from cartridges.datasets import CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64

LORA_RANK = os.environ.get("LORA_RANK", "192")
LORA_RANK = list(map(int, LORA_RANK.split(",")))

LR = os.environ.get("LR", "2e-4")
LR = list(map(float, LR.split(",")))

# TOKENS_TO_LR = {8192: 0.001, 4096: 0.005, 2048: 0.005, 1024: 0.001, 512: 0.05, 256: 0.05, 128: 0.05}


configs = [
    TrainConfig(
        name=FormatStringVariable(
            f"{file_name}_lr{{lr}}_rank{{model.peft.r}}"
        ),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
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
        dataset=CartridgeDatasetLatest.Config(
            data_sources=[
                ("hazy-research/Cartridges/generate_mtob_simple_latex_s5_n65536:v0", None),
                ("hazy-research/Cartridges/generate_mtob_simple_latex_s5_n65536:v1", None),
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
        loss_type="logits",
        epochs=1,
        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "mtob"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        distributed_backend=(
            "gloo" 
        ),
        global_batch_size=bs,
        local_batch_size=2,
        use_batch_sampler=True,
    )
    for rank in LORA_RANK
    for lr in LR
]

if __name__ == "__main__":
    pydrantic.main(configs)
