

import os
from pathlib import Path
import pydrantic


from capsules.optim import CosWithWarmup
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from capsules.config import HFModelConfig
from capsules.datasets import ( CapsuleDatasetLatest )
from capsules.utils import WandBConfig
from capsules.tasks.finance import ( FinanceBenchEvalDataset, FinanceBenchGenerateDataset, FinanceBenchMemorizationDataset )
from capsules.tasks.finance import FinanceBenchDocumentStructuredConfig


file_name = Path(__file__).stem
configs = []
bs = 64


# Experiment: Cartridge
dataset_sources_dict = {

    # "AMD_2022_10K": [
    # ],
    # "PEPSICO_2022_10K": [
    # ],
    # "BOEING_2022_10K": [
    #     ("/data/simran/2025-05-12-21-28-39-m05d11_compose_generate/bb636b6f-8bb2-48a7-a2e6-30d4104f9fc0/artifact/dataset.pkl", None)
    # ],
    "AMERICANEXPRESS_2022_10K": [
        ("/data/simran/2025-05-12-21-44-50-m05d11_compose_generate/9b1e2602-47bc-4030-9100-15097ca4de0e/artifact/dataset.pkl", None)
    ],

}


configs = []


for doc_name, dataset_sources in dataset_sources_dict.items():
        
    # for num_tokens, lr in [(2048, 0.01), (4096, 0.005), (8192, 0.005)]:

    # for num_tokens, lr in [(512, 0.05), (1024, 0.05), (2048, 0.01), (4096, 0.005), (8192, 0.005)]:

    for num_tokens, lr in [(2048, 0.05), (4096, 0.05), (8192, 0.05)]:

        configs.append(

            TrainConfig(

                # Data and initialization
                name=f"{file_name}_nt{num_tokens}_auto_lr{lr}_{doc_name}",
                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),
                tokenizer="meta-llama/Llama-3.2-3B-Instruct",
                lr=lr,

                dataset=CapsuleDatasetLatest.Config(
                    data_sources=dataset_sources,
                    is_wandb=True,
                    label_type="logits",
                    top_k_logits=20,
                ),

                context=FinanceBenchDocumentStructuredConfig(
                    doc_names=[doc_name], 
                ),

                generate_max_new_tokens=512,
                kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                    max_tokens=num_tokens
                ),

                # Evals and generations
                generate_every_n_steps=1024,
                generate_datasets=[
                    GenerateDatasetConfig(
                        name_for_wandb="finance-ppl-gt",
                        dataset=FinanceBenchGenerateDataset.Config(
                            doc_names=[doc_name],
                        ),
                    )
                ],

                eval_every_n_steps = 256,
                eval_datasets=[
                    EvalDatasetConfig(
                        name_for_wandb="finance-ppl-gt",
                        local_batch_size=8,
                        dataset=FinanceBenchEvalDataset.Config(
                            doc_names=[doc_name],
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
                            doc_names=[doc_name],
                            cot=False,
                            label_type="tokens",
                            max_questions=10,
                            data_sources=[],  # ignore this arg
                        ),
                        only_eval_rank_0=True,
                    )
                ],


                # Training setup 
                loss_type="logits",
                save_every_n_steps=256,
                epochs=1,
                
                wandb=WandBConfig(
                    project="capsules",
                    tags=["financebench", "auto_train", f"doc{doc_name}"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
                global_batch_size=bs,
                local_batch_size=4,

                distributed_backend="gloo",
            )
        )


if __name__ == "__main__":
   for config in configs:
       pydrantic.main([config])


