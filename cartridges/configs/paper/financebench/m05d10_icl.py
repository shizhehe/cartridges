import os
from pathlib import Path
import pydrantic

from capsules.clients.tokasaurus import TokasaurusClient
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.generate_baseline import GenerateBaselineConfig, ICLBaseline
from capsules.train import GenerateDatasetConfig
from capsules.utils import WandBConfig
from capsules.config import HFModelConfig
from capsules.datasets import ( CapsuleDatasetLatest )

from capsules.tasks.finance import ( FinanceBenchEvalDataset, FinanceBenchGenerateDataset, FinanceBenchMemorizationDataset )
from capsules.tasks.finance import FinanceBenchContextConfig, FinanceBenchDocumentStructuredConfig



file_name = Path(__file__).stem

model_path = "meta-llama/Llama-3.2-3B-Instruct"

tokasaurus_client = TokasaurusClient.Config(
    url="https://hazyresearch--tksrs-batch-main-llama-3b-1xh100-min0-serve.modal.run/v1",
    use_modal_endpoint=True,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
)

SYSTEM_PROMPT_TEMPLATE = f"""Please reference the information below to answer the questions.

<reference>
{{content}}
</reference>
"""


configs = []
bs = 1

for DOC_NAME in [
    "AMD_2022_10K", 
    "BOEING_2022_10K", 
    "AMERICANEXPRESS_2022_10K", 
    "PEPSICO_2022_10K"
]:
    
    for num_tokens, lr in [(131702, 0.00)]:

        configs.append(

            TrainConfig(
                name=f"{file_name}_{DOC_NAME}_nt{num_tokens}",

                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),

                dataset=CapsuleDatasetLatest.Config(
                    data_sources=[("/data/simran/2025-05-10-10-59-39-m05d10_generate/6fdf01a7-3713-4d29-9d98-f7fbb9719988/artifact/dataset.pkl", None)], # Garbage
                    is_wandb=True,
                    label_type="logits",
                    top_k_logits=20,
                ),

                context=FinanceBenchDocumentStructuredConfig(
                    doc_names=[DOC_NAME], 
                ),

                eval_every_n_steps=64,
                eval_datasets=[
                    EvalDatasetConfig(
                        name_for_wandb="finance-ppl-gt",
                        local_batch_size=4,
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
                        local_batch_size=4,
                        dataset=FinanceBenchMemorizationDataset.Config(
                            doc_names=[DOC_NAME],
                            cot=False,
                            label_type="tokens",
                            max_questions=10,
                            data_sources=[],  # ignore this arg
                        ),
                        only_eval_rank_0=True,
                    )
                ],

                generate_max_new_tokens=512,
                kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                    max_tokens=num_tokens
                ),
                
                loss_type="logits",
                save_every_n_steps=128,
                epochs=0,
                lr=lr,

                save_to_wandb = False,

                wandb=WandBConfig(
                    project="capsules",
                    tags=["financebench", "icl", f"doc{DOC_NAME}"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
                global_batch_size=bs,
                local_batch_size=1,
            )
        )

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])


