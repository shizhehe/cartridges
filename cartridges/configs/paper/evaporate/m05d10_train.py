import os
from pathlib import Path
import pydrantic


from capsules.optim import CosWithWarmup
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from capsules.config import HFModelConfig
from capsules.datasets import ( CapsuleDatasetLatest )
from capsules.utils import WandBConfig


from capsules.tasks.fda import EvaporateContextConfig, EvaporateMultipleChoiceGenerateDataset, EvaporateEvalDataset



file_name = Path(__file__).stem
configs = []
bs = 64


# Experiment: Cartridge V1 SimplePromptSampler
dataset_sources_dict = {
    "K152386.txt": [
        ("/home/user/capsules/2025-05-11-04-25-03-m05d10_generate/41c19c8b-7338-440b-9f05-ae1825325c79/artifact/dataset.pkl", None),
        # ("/home/user/dataset_K152386.pkl", None)
    ],
    # "K182513.txt": [
    #     ("/data/simran/dataset_K182513.pkl", None)
    # ],
    # "K181324.txt": [
    #     ("/data/simran/dataset_K181324.pkl", None)
    # ],
    # "K173887.txt": [
    #     ("2025-05-10-14-12-36-m05d10_generate/cdb19e9f-e9a9-4566-a432-77bcf237423f/artifact/dataset.pkl", None)
    # ]
}

configs = []


for doc_name, dataset_sources in dataset_sources_dict.items():

    for num_tokens, lr in [(1024, 0.05), (1024, 0.1)]:

        configs.append(

            TrainConfig(

                # Data and initialization
                name=f"{file_name}_nt{num_tokens}_auto_lr{lr}_{doc_name}",
                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),
                lr=lr,
                tokenizer="meta-llama/Llama-3.2-3B-Instruct",

                dataset=CapsuleDatasetLatest.Config(
                    data_sources=dataset_sources,
                    is_wandb=True,
                    label_type="logits",
                    top_k_logits=20,
                ),

                context=EvaporateContextConfig(
                    doc_id=doc_name, 
                    max_tokens_per_section=num_tokens,
                ),

                generate_max_new_tokens=512,
                kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(
                    max_tokens=num_tokens
                ),

                # Evals and generations
                generate_every_n_steps=64,
                generate_datasets=[
                    GenerateDatasetConfig(
                        dataset= EvaporateMultipleChoiceGenerateDataset.Config(
                            doc_id=doc_name, 
                            cot=False,
                        ),
                        name_for_wandb=f"evaporate_{doc_name}_mc",
                    ),
                ],

                eval_every_n_steps=64,
                eval_datasets=[
                    EvalDatasetConfig(
                        name_for_wandb=f"evaporate_mc",
                        local_batch_size=16,
                        dataset=EvaporateEvalDataset.Config(
                            doc_id=doc_name,
                            max_questions=256,
                            label_type="tokens",
                            data_sources=[]  # ignore this arg
                        )
                    )
                ],


                # Training setup 
                loss_type="logits",
                save_every_n_steps=128,
                epochs=1,
                
                wandb=WandBConfig(
                    project="capsules",
                    tags=["evaporate", "auto_train", f"doc{doc_name}"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
                global_batch_size=bs,
                local_batch_size=4,

                distributed_backend="nccl",
            )
        )


if __name__ == "__main__":
   for config in configs:
       pydrantic.main([config])


