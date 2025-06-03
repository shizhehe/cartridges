import os
from pathlib import Path

import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.kv_initialization.strategies.random import KVFromRandomText
from capsules.optim import CosWithWarmup
from capsules.tasks.codehop.code_hop_synth import CodeHopSynthConfig
from capsules.tasks.codehop.generate_dataset import CodeHopGenerateDataset
from capsules.tasks.finance import (
    FinanceBenchEvalDataset,
)
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.tasks.mtob import mtob_eval_datasets, mtob_generate_datasets
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import (
    CapsuleDatasetLatest,
)

from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64

configs = []
if True:

    # for num_tokens, lr in [(6144, 0.01), (6144, 0.02,), (6144, 0.04), (6144, 0.005), (6144, 0.1)]:
    for num_tokens in [512]:
        for lr in [0.01]:
            sources = [
                (
                    "hazy-research/capsules/m05d09_generate_codehop_2_files_0a8a11_question_n16384_cot0.2:v0",
                    None,
                )
            ]
            configs.append(
                TrainConfig(
                    name=f"{file_name}_nt{num_tokens}",
                    model=HFModelConfig(
                        pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
                    ),
                    dataset=CapsuleDatasetLatest.Config(
                        data_sources=sources,
                        is_wandb=True,
                        label_type="logits",
                        top_k_logits=20,
                    ),
                    generate_every_n_steps=16,
                    generate_datasets=[
                        GenerateDatasetConfig(
                            dataset=CodeHopGenerateDataset.Config(
                                code_hop_config=CodeHopSynthConfig(
                                    seed=34,
                                    num_files=2,
                                    num_methods_per_file=5,
                                    method_name_length=4,
                                    deepest_call_chain=4,
                                    input_vocab_size=4,
                                    output_vocab_size=4,
                                    function_name_vocab_size=50,
                                )
                            ),
                            name_for_wandb="code_hop",
                        )
                    ],
                    eval_every_n_steps=64,
                    eval_datasets=[],
                    generate_max_new_tokens=64,
                    # kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                    #     max_tokens=num_tokens
                    # ),
                    kv_cache_initializer=KVFromRandomText.Config(max_tokens=num_tokens),
                    loss_type="logits",
                    save_every_n_steps=128,
                    epochs=4,
                    lr=lr,
                    lr_scheduler=CosWithWarmup.Config(max_steps=5000, warmup_steps=100),
                    wandb=WandBConfig(
                        project="capsules",
                        tags=["cache_tuning", "development"],
                        entity="hazy-research",
                    ),
                    output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
                    global_batch_size=bs,
                    local_batch_size=2,
                    # distributed_backend="nccl",
                )
            )

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])
