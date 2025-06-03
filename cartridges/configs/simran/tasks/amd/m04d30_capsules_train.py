import os
from pathlib import Path
import pydrantic

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.tasks.finance import ( FinanceBenchEvalDataset, FinanceBenchGenerateDataset, FinanceBenchMemorizationDataset )
from capsules.tasks.mmlu import MMLUEvalDataset
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import ( CapsuleDatasetLatest )
from capsules.utils import WandBConfig


file_name = Path(__file__).stem


configs = []
DOC_NAME = "AMD_2022_10K"
bs = 64


dataset_sources = [
    # ("/data/simran/capsules/outputs/2025-05-02-13-32-59-m0430_generate_v0/7263b2c1-c5ad-4ff3-9027-41dac5d0e87f/artifact/dataset.pkl", None,),
    # ("/data/simran/capsules/outputs/2025-05-02-15-25-31-m05d01_generate_simpleqa_v0/37dbd19a-bd03-4540-9dff-fc88f66395fa/artifact/dataset.pkl", None,),
    ("/data/simran/capsules/outputs/2025-05-02-20-21-39-m05d01_generate_memorize_subcontext_v0/89925dfa-f385-41d6-adf0-ddcab78a22e2/artifact/dataset.pkl", None,),
    # ("/data/simran/capsules/outputs/2025-05-03-12-33-57-m05d01_generate_simpleqa_windowed_v0/03611a27-7c29-4959-8458-d3934ac63a29/artifact/dataset.pkl", None,),
    ("/data/simran/capsules/outputs/2025-05-04-10-06-29-m05d01_generate_simpleqa_subcontext_v0/54c1c92f-2c2e-403d-9603-ad78f146191e/artifact/dataset.pkl", None,),
]

# dataset_mix = []
# for dataset in dataset_sources:
#     dataset = dataset[0]
#     if "memorize" in dataset:
#         dataset_mix.append("memorize")
#     elif "simpleqa" in dataset:
#         dataset_mix.append("simpleqa")
#     else:
#         dataset_mix.append("memorize")
# dataset_mix = "_".join(dataset_mix)

dataset_mix = "auto_baseline"


configs = []

generate_evals, ppl_evals = get_evals(
    FinanceEvals,
    DOC_NAME,
    num_samples=16,  # RE: it's silly we have to specify this
    version_tag="v1",
    batch_size=16,
)

lr = 5e-2
for num_tokens, lr in [(4096, 0.05)]:
    configs.append(

        TrainConfig(
            name=f"{file_name}_{DOC_NAME}_nt{num_tokens}_data{dataset_mix}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
            ),

            dataset=CapsuleDatasetLatest.Config(
                data_sources=dataset_sources,
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
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

            loss_type="logits",
            save_every_n_steps=128,
            epochs=1,
            lr=lr,
            wandb=WandBConfig(
                project="capsules",
                tags=["cache_tuning", "development"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=4,
        )
    )

if __name__ == "__main__":
    config_idx = 0
    selected_config = configs[config_idx]
    pydrantic.main([selected_config])


