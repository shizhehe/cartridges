import os
from pathlib import Path
import pydrantic


from cartridges.optim import CosWithWarmup
from cartridges.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from cartridges.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import ( CartridgeDatasetLatest )
from cartridges.utils import WandBConfig
from cartridges.tasks.finance import ( FinanceBenchEvalDataset, FinanceBenchGenerateDataset, FinanceBenchMemorizationDataset )
from cartridges.tasks.finance import FinanceBenchDocumentStructuredConfig


file_name = Path(__file__).stem
configs = []
bs = 64


is_w_deps = True


if not is_w_deps:

    dataset_sources_dict = {
        "AMD_2022_10K": [
            ("/data/simran/2025-05-12-19-22-53-m05d10_generate/ad8d5a77-284d-4b81-b556-e880381737df/artifact/dataset.pkl", None)
        ],
        "BOEING_2022_10K": [
            ("/data/simran/2025-05-12-19-37-33-m05d10_generate/c9398462-327c-4eee-a662-e7e05173b348/artifact/dataset.pkl", None)
        ],
        "AMERICANEXPRESS_2022_10K": [
            ("/data/simran/2025-05-12-19-51-00-m05d10_generate/ed80f387-7dc5-4a45-8175-2b91430647e8/artifact/dataset.pkl", None)
        ],
        "PEPSICO_2022_10K": [
            ("/data/simran/2025-05-12-20-04-03-m05d10_generate/62be6c7d-7c32-4163-8df4-f4b79e6320cb/artifact/dataset.pkl", None)
        ]
    }

else:

    dataset_sources_dict = {
        # Add global dependency tags
        # "AMD_2022_10K": [
        #     ("/data/simran/2025-05-11-16-51-27-m05d11_compose_generate/9ceca541-acc0-4d1a-9e31-d7eaa51b0ecf/artifact/dataset.pkl", None)
        # ],
        # "PEPSICO_2022_10K": [
        #     ("/data/simran/2025-05-11-17-05-13-m05d11_compose_generate/0d303ad7-a4c6-4a79-8d09-13ed36cdd0aa/artifact/dataset.pkl", None)   
        # ],
        # "BOEING_2022_10K": [
        #     ("/data/simran/2025-05-11-20-21-26-m05d11_compose_generate/c23df0d6-34ac-42ca-b088-3fab6ad10c25/artifact/dataset.pkl", None)
        # ],
        # "AMERICANEXPRESS_2022_10K": [
        #     ("/data/simran/2025-05-11-20-34-24-m05d11_compose_generate/970db361-a76b-4e01-bbcb-c4e76d7e778f/artifact/dataset.pkl", None)
        # ],
    }


configs = []


for doc_name, dataset_sources in dataset_sources_dict.items():
        
    # for num_tokens, lr in [(2048, 0.01), (4096, 0.005), (8192, 0.005)]:

    # for num_tokens, lr in [(512, 0.05), (1024, 0.05), (2048, 0.01), (4096, 0.005), (8192, 0.005)]:

    for num_tokens, lr in [(512, 0.05), (1024, 0.05), (2048, 0.05), (4096, 0.05), (8192, 0.05)]:

        configs.append(

            TrainConfig(

                # Data and initialization
                name=f"{file_name}_nt{num_tokens}_auto_lr{lr}_{doc_name}_deps{is_w_deps}",
                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),
                tokenizer="meta-llama/Llama-3.2-3B-Instruct",
                lr=lr,

                dataset=CartridgeDatasetLatest.Config(
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
                    project="cartridges",
                    tags=["financebench", "auto_train", f"doc{doc_name}"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
                global_batch_size=bs,
                local_batch_size=4,

                distributed_backend="gloo",
            )
        )


if __name__ == "__main__":
   for config in configs:
       pydrantic.main([config])


