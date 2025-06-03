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
from capsules.tasks.codehop.code_hop_dataset import CodeHopDataset
from capsules.tasks.codehop.code_hop_synth import CodeHopSynthConfig
from capsules.tasks.codehop.generate_dataset import CodeHopGenerateDataset
from capsules.tasks.finance import (
    FinanceBenchEvalDataset,
)
from capsules.tasks.mmlu import MMLUEvalDataset, MMLUGenerateDataset, mmlu_subset
from capsules.tasks.mtob import mtob_eval_datasets, mtob_generate_datasets
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import (
    CapsuleDatasetLatest,
)

from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 16

syth_config = CodeHopSynthConfig(
    seed=34,
    num_files=2,
    num_methods_per_file=10,
    method_name_length=4,
    deepest_call_chain=4,
    input_vocab_size=6,
    output_vocab_size=4,
    function_name_vocab_size=50,
)


configs = []
if True:
    num_tokens = 256

    configs.append(
        TrainConfig(
            name=f"{file_name}_nt{num_tokens}_.1ema",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct"
            ),
            dataset=CodeHopDataset.Config(
                code_hop_config=syth_config,
                include_header=True,
            ),
            generate_every_n_steps=2,
            generate_datasets=[
                GenerateDatasetConfig(
                    dataset=CodeHopGenerateDataset.Config(code_hop_config=syth_config),
                    name_for_wandb="code_hop",
                )
            ],
            eval_every_n_steps=64,
            eval_datasets=[
                # *mtob_eval_datasets(
                #     local_batch_size=16,
                # ),
                # EvalDatasetConfig(
                #     name_for_wandb="mmlu",
                #     local_batch_size=16,
                #     dataset=MMLUEvalDataset.Config(num_samples=128),
                # ),
            ],
            generate_max_new_tokens=4,
            kv_cache_initializer=KVFromRandomText.Config(max_tokens=num_tokens),
            loss_type="online",
            save_every_n_steps=128,
            epochs=128,
            online_model=False,
            lr=0.01,
            # lr_scheduler=CosWithWarmup.Config(max_steps=5000, warmup_steps=100),
            wandb=WandBConfig(
                project="capsules",
                tags=["cache_tuning", "development"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=8,
            # distributed_backend="nccl",
        )
    )

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])
