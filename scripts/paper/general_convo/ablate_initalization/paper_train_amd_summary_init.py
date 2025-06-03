import os
from pathlib import Path
import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.configs.paper.amd.ablate_initalization.paper_train_amd_common import (
    get_config,
)
from cartridges.kv_initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.tasks.finance import (
    FinanceBenchEvalDataset,
    FinanceBenchGenerateDataset,
    FinanceBenchMemorizationDataset,
)
from cartridges.tasks.mmlu import MMLUEvalDataset
from cartridges.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDatasetLatest
from cartridges.utils import WandBConfig


file_name = Path(__file__).stem


configs = [
    get_config(
        mix_name="structuring",
        experiment_name="paper_train_amd_summary_init",
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        learning_rate=0.02,
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
            max_tokens=4096,
        ),
    )
]

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])
