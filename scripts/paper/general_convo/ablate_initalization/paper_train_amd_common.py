import os
from pathlib import Path
import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.tasks.finance import (
    FinanceBenchEvalDataset,
    FinanceBenchGenerateDataset,
    FinanceBenchMemorizationDataset,
)
from cartridges.tasks.mmlu import MMLUEvalDataset, mmlu_subset
from cartridges.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDatasetLatest
from cartridges.utils import WandBConfig


file_name = Path(__file__).stem


configs = []
DOC_NAME = "AMD_2022_10K"
bs = 64

configs = []

generate_evals, ppl_evals = get_evals(
    FinanceEvals,
    DOC_NAME,
    num_samples=16,  # RE: it's silly we have to specify this
    version_tag="v1",
    batch_size=16,
)


sources = {
    "structuring": [
        (
            "hazy-research/Cartridges/paper_generate_amd_docAMD_2022_10K_n8192_slicestructuring:v0",
            None,
        )
    ],
    "all": [
        (
            f"hazy-research/Cartridges/paper_generate_amd_docAMD_2022_10K_n8192_slice{slice_}:v0",
            None,
        )
        for slice_ in [
            "structuring",
            "summarization",
            "aggregation",
            "question",
            "use_case",
        ]
    ],
}


def get_config(
    mix_name, experiment_name, model_name, learning_rate: float, kv_cache_initializer
):
    return TrainConfig(
        name=f"{experiment_name}_{DOC_NAME}_auto_data_{mix_name}",
        model=HFModelConfig(
            pretrained_model_name_or_path=model_name,
        ),
        dataset=CartridgeDatasetLatest.Config(
            data_sources=sources["structuring"],
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        generate_every_n_steps=1024,
        generate_datasets=[
            GenerateDatasetConfig(
                name_for_wandb="finance-ppl-gt",
                dataset=FinanceBenchGenerateDataset.Config(
                    doc_names=[DOC_NAME],
                ),
            ),
            mmlu_subset(),
            *generate_evals,
        ],
        eval_every_n_steps=64,
        eval_datasets=[
            EvalDatasetConfig(
                name_for_wandb="finance-ppl-gt",
                local_batch_size=8,
                dataset=FinanceBenchEvalDataset.Config(
                    doc_names=[DOC_NAME],
                    cot=False,
                    label_type="tokens",
                    data_sources=[],  # ignore this arg
                ),
                only_eval_rank_0=True,
            ),
            EvalDatasetConfig(
                name_for_wandb="finance-memorization",
                local_batch_size=8,
                dataset=FinanceBenchMemorizationDataset.Config(
                    doc_names=[DOC_NAME],
                    cot=False,
                    label_type="tokens",
                    max_questions=10,
                    data_sources=[],  # ignore this arg
                ),
                only_eval_rank_0=True,
            ),
            *ppl_evals,
        ],
        generate_max_new_tokens=512,
        kv_cache_initializer=kv_cache_initializer,
        loss_type="logits",
        save_every_n_steps=128,
        epochs=1,
        lr=learning_rate,
        wandb=WandBConfig(
            project="cartridges",
            tags=["cache_tuning", "development"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        global_batch_size=bs,
        local_batch_size=1,
    )


if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])
