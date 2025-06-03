import os
from pathlib import Path

from cartridges.models.llama import LlamaForCausalLM
import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.initialization.strategies.first_n_tokens import (
    KVCacheInitFromFirstNTokensOfContext,
)
from cartridges.initialization.strategies.random import KVFromRandomText
from cartridges.optim import CosWithWarmup
from cartridges.tasks.codehop.code_hop_dataset import CodeHopDataset
from cartridges.tasks.codehop.code_hop_synth import CodeHopSynthConfig
from cartridges.tasks.codehop.generate_dataset import CodeHopGenerateDataset
from cartridges.tasks.finance import (
    FinanceBenchEvalDataset,
)
from cartridges.tasks.mmlu import MMLUEvalDataset, MMLUGenerateDataset, mmlu_subset
from cartridges.tasks.mtob import mtob_eval_datasets, mtob_generate_datasets
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import (
    CartridgeDatasetLatest,
)

from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64

syth_config = CodeHopSynthConfig(
    seed=42,
    num_files=1,
    num_methods_per_file=5,
    method_name_length=4,
    deepest_call_chain=4,
    input_vocab_size=4,
    output_vocab_size=8,
    function_name_vocab_size=50,
)


configs = []
if True:
    num_tokens = 256

    configs.append(
        TrainConfig(
            name=f"{file_name}_nt{num_tokens}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
                # model_cls=LlamaForCausalLM,
                # attn_implementation="einsum",
            ),
            dataset=CodeHopDataset.Config(
                code_hop_config=syth_config,
                include_header=True,
            ),
            generate_every_n_steps=4,
            generate_datasets=[
                GenerateDatasetConfig(
                    dataset=CodeHopGenerateDataset.Config(
                        code_hop_config=syth_config,
                    ),
                    name_for_wandb="code_hop",
                    batch_size=1,
                ),
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
            generate_max_new_tokens=8,
            kv_cache_initializer=KVFromRandomText.Config(max_tokens=num_tokens),
            loss_type="online",
            save_every_n_steps=128,
            epochs=64,
            online_model=False,
            lr=0.01,
            # lr_scheduler=CosWithWarmup.Config(
            #     max_steps=100, warmup_steps=5, warmup_min_lr=0.001
            # ),
            wandb=WandBConfig(
                project="cartridges",
                tags=["train", "codehop"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=4,
            # distributed_backend="nccl",
        )
    )

if __name__ == "__main__":
    config_idx = int(os.environ.get("CFG_IDX", default=0))
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])
