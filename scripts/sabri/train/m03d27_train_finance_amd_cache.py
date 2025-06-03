import os
from pathlib import Path

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.tasks.finance import FinanceBenchEvalDataset, FinanceBenchGenerateDataset, L3bFinanceBenchEvalDataset, L3bFinanceBenchGenerateDataset
from cartridges.tasks.mmlu import MMLUEvalDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset

from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

configs = []
DOC_NAME = "AMD_2022_10K"
NUM_TOKENS = 4096 # [2048, 4096, 2048 * 4]
GLOBAL_BATCH_SIZE = 64

configs = []
configs.append(TrainConfig(
    name=FormatStringVariable(f"{file_name}_{DOC_NAME}_nt{{kv_cache_initializer.max_tokens}}_bs{{global_batch_size}}"),
    model=HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
    ),
    dataset=CartridgeDataset.Config(
        data_sources=[
            ("hazy-research/Cartridges/m03d27_gen_simple_data_amd_2022_10k:v0", None),
        ],  
        is_wandb=True,
        label_type="logits",
        top_k_logits=20,
    ),
    generate_every_n_steps=64,
    generate_datasets=[
        GenerateDatasetConfig(
            name_for_wandb="finance-bench",
            dataset=L3bFinanceBenchGenerateDataset.Config(
                doc_names=[DOC_NAME],
            )
        ),
        GenerateDatasetConfig(
            name_for_wandb="finance-bench-gt",
            dataset=FinanceBenchGenerateDataset.Config(
                doc_names=[DOC_NAME],
            )
        ),
    ],
    eval_every_n_steps=2,
    eval_datasets=[
        EvalDatasetConfig(
            name_for_wandb="finance-ppl",
            local_batch_size=16,
            dataset=L3bFinanceBenchEvalDataset.Config(
                doc_names=[DOC_NAME],
                cot=False,
                label_type="tokens",
                data_sources=[]  # ignore this arg
            ),
            only_eval_rank_0=True
        ),
        EvalDatasetConfig(
            name_for_wandb="finance-ppl-gt",
            local_batch_size=16,
            dataset=FinanceBenchEvalDataset.Config(
                doc_names=[DOC_NAME],
                cot=False,
                label_type="tokens",
                data_sources=[]  # ignore this arg
            ),
            only_eval_rank_0=True,
        ),
        EvalDatasetConfig(
            name_for_wandb="anthropic-questions",
            local_batch_size=16,
            dataset=CartridgeDataset.Config(
                data_sources=[
                    ("hazy-research/Cartridges/m03d27_amd10k_sonnet37_manual_eval:v1", None)
                ],
                is_wandb=True,
                label_type="tokens",
            ),
        ),

        # EvalDatasetConfig(
        #     name_for_wandb="mmlu",
        #     local_batch_size=16,
        #     dataset=MMLUEvalDataset.Config(num_samples=256),
        # ),
    ],
    generate_max_new_tokens=1024,
    kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
        max_tokens=NUM_TOKENS
    ),
    loss_type="logits",
    save_every_n_steps=64,
    epochs=1,
    lr=5e-4,
    wandb=WandBConfig(
        project="cartridges",
        tags=["cache_tuning", "development"],
        entity="hazy-research",
    ),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    global_batch_size=GLOBAL_BATCH_SIZE,
    local_batch_size=1,
))

if __name__ == "__main__":
    pydrantic.main(configs)
