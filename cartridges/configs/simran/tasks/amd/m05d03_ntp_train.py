import os
from pathlib import Path
import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.tasks.finance import ( FinanceBenchEvalDataset, FinanceBenchGenerateDataset, FinanceBenchMemorizationDataset, FinanceBenchContextConfig )
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
# from capsules.datasets import ( CapsuleDatasetLatest )
from capsules.datasets import CapsuleDataset
from capsules.utils import WandBConfig


file_name = Path(__file__).stem


configs = []
DOC_NAME = "AMD_2022_10K"
gbs = 16
lbs = 2
dataset_mix = "ntp"

configs = []

generate_evals, ppl_evals = get_evals(
    FinanceEvals,
    DOC_NAME,
    num_samples=16,  
    version_tag="v1",
    batch_size=16,
)


num_train_tokens = 1_000_000_000
max_optimizer_steps = (num_train_tokens // gbs) 

# (4096, 0.05), (2048, 0.05), (1024, 0.05), (512, 0.05), 
for num_tokens, lr in [(4096, 0.005), (2048, 0.01), (1024, 0.05), (512, 0.05), (8192, 0.005)]:
    configs.append(

        TrainConfig(
            name=f"{file_name}_{DOC_NAME}_nt{num_tokens}_data{dataset_mix}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),

            dataset=CapsuleDataset.Config(
                data_sources=[(
                    "hazy-research/capsules/m04d03_generate_ntp_3b_docAMD_2022_10K_sections4_npreview64_max1024_32000:v0", None,
                )],
                is_wandb=True,
                label_type="tokens",
                # top_k_logits=20,
            ),

            context=FinanceBenchContextConfig(
                doc_names=[DOC_NAME], 
            ),

            eval_every_n_steps=64,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb="finance-ppl-gt",
                    local_batch_size=16,
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
                    local_batch_size=16,
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
            kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                max_tokens=num_tokens
            ),

            loss_type="tokens",
            save_every_n_steps=128,
            epochs=1,
            max_optimizer_steps=max_optimizer_steps,
            lr=lr,
            wandb=WandBConfig(
                project="capsules",
                tags=["cache_tuning", "development"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=gbs,
            local_batch_size=lbs,

            distributed_backend="gloo",
        )
    )

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])


