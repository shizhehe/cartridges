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


is_w_deps = True


if not is_w_deps:

    dataset_sources_dict = {
        # "AMD_2022_10K": [
        #     ("2025-05-14-14-40-39-m05d11_compose_generate/9e225208-780d-4af5-a49a-96e9a219b9fa/artifact/dataset.pkl", None),
        # ],
        # "BOEING_2022_10K": [
        #     ("2025-05-14-13-51-29-m05d11_compose_generate/dd3f336c-430a-4667-ab7c-946ccb661176/artifact/dataset.pkl", None),
        # ],
        # "AMERICANEXPRESS_2022_10K": [
        #     ("2025-05-14-14-07-38-m05d11_compose_generate/e252588f-6018-476e-aacf-b58d9621883d/artifact/dataset.pkl", None),
        # ],
        "PEPSICO_2022_10K": [
            ("2025-05-14-14-25-22-m05d11_compose_generate/9aabcab3-2a2a-4946-999f-3f9147be231e/artifact/dataset.pkl", None),
        ]
    }

else:

    dataset_sources_dict = {
        # # Add global dependency tags
        # "AMD_2022_10K": [
        #     ("/data/simran/2025-05-14-14-45-41-m05d11_compose_generate/cb2e0b01-743d-4ad4-8c81-34c4a11b9f93/artifact/dataset.pkl", None),
        # ],
        # "BOEING_2022_10K": [
        #     ("/data/simran/2025-05-14-13-58-52-m05d11_compose_generate/354e7579-6b7d-424b-b044-36caeae9d557/artifact/dataset.pkl", None),
        # ],
        # "AMERICANEXPRESS_2022_10K": [
        #     ("/data/simran/2025-05-14-14-14-29-m05d11_compose_generate/72de5801-a3b7-4cf4-af19-3b6b2f3faa40/artifact/dataset.pkl", None),
        # ],
        "PEPSICO_2022_10K": [
            ("/data/simran/2025-05-14-14-29-49-m05d11_compose_generate/becb2333-0266-459d-a5b1-145bead8bfb4/artifact/dataset.pkl", None),
        ],
    }


configs = []


for doc_name, dataset_sources in dataset_sources_dict.items():
        
    for num_tokens, lr in [(2048, 0.05), (4096, 0.05)]: # (512, 0.05),  (8192, 0.005), (1024, 0.05), 

        configs.append(

            TrainConfig(

                # Data and initialization
                name=f"{file_name}_nt{num_tokens}_auto_lr{lr}_{doc_name}_deps{is_w_deps}",
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


