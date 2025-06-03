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
from cartridges.tasks.finance import FinanceBenchContextConfig, FinanceBenchDocumentStructuredConfig


file_name = Path(__file__).stem
configs = []
bs = 64


# Experiment: Cartridge V1 SimplePromptSampler
dataset_sources_dict = {
    "AMERICANEXPRESS_2022_10K": [
        ("/data/simran/2025-05-10-11-42-57-m05d10_generate/b8e9ccb4-c0d5-4e4a-b9d6-1d477b2a0f04/artifact/dataset.pkl", None)
    ],
}


configs = []
size = 4096

for doc_name, dataset_sources in dataset_sources_dict.items():

    for num_tokens, lr in [(size, 0.001), (size, 0.005), (size, 0.01), (size, 0.05), (size, 0.08), (size, 0.1)]: 

        configs.append(

            TrainConfig(

                # Data and initialization
                name=f"{file_name}_nt{num_tokens}_lr{lr}_{doc_name}",
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

                eval_every_n_steps=256,
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
                save_every_n_steps=128,
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


