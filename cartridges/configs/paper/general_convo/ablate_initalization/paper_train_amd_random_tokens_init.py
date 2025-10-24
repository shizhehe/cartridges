import os
from pathlib import Path
import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.configs.paper.amd.ablate_initalization.paper_train_amd_common import (
    get_config,
)
from capsules.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from capsules.kv_initialization.strategies.random import KVFromRandomText
from capsules.tasks.finance import (
    FinanceBenchEvalDataset,
    FinanceBenchGenerateDataset,
    FinanceBenchMemorizationDataset,
)
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDatasetLatest
from capsules.utils import WandBConfig


file_name = Path(__file__).stem


configs = [
    get_config(
        mix_name="all",
        experiment_name="paper_train_amd_random_tokens_init",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        learning_rate=0.02,
        kv_cache_initializer=KVFromRandomText.Config(
            max_tokens=4096,
        ),
    )
]

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])
