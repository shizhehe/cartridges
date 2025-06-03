import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.models.llama import LlamaForCausalLM
from cartridges.optim import CosWithWarmup
from cartridges.tasks.longhealth import (
    LongHealthEvalDataset,
    LongHealthMultipleChoiceGenerateDataset,
)
from cartridges.tasks.longhealth.context import LongHealthStructuredContextConfig
from cartridges.tasks.mtob import mtob_generate_datasets
from cartridges.tasks.mtob.context import MTOBNoStructuredContext
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64

# NUM_TOKENS = os.environ.get("NUM_TOKENS", "2048")
# NUM_TOKENS = list(map(int, NUM_TOKENS.split(",")))

NUM_TOKENS = [8192]

TOKENS_TO_LR = {
    8192: 0.001,
    4096: 0.005,
    2048: 0.005,
    1024: 0.001,
    512: 0.05,
    256: 0.05,
    128: 0.05,
}


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
        lr=0.005,
        dataset=CartridgeDatasetLatest.Config(
            data_sources=[
                (
                    "hazy-research/Cartridges/m05d05_generate_mtob_latex_structuring_summarization_aggregation_question_use_case_creative_65536:v0",
                    None,
                ),
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
        ],
        eval_every_n_steps=256,
        eval_datasets=[],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=num_tokens
        ),
        loss_type="logits",
        epochs=2,
        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "mtob"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        distributed_backend=(
            "gloo"
            if socket.gethostname().startswith("mk-ix-")
            or socket.gethostname() in ("22b3cd4f", "g425")
            else "nccl"
        ),
        global_batch_size=bs,
        local_batch_size=2,
        use_batch_sampler=True,
    )
    for num_tokens in NUM_TOKENS
]

if __name__ == "__main__":
    pydrantic.main(configs)
