import os
from pathlib import Path
import pydrantic

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.generate_baseline import GenerateBaselineConfig, ICLBaseline
from cartridges.train import GenerateDatasetConfig
from cartridges.utils import WandBConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import ( CartridgeDatasetLatest )

from cartridges.tasks.fda import EvaporateContextConfig, EvaporateMultipleChoiceGenerateDataset, EvaporateEvalDataset



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
    "K152386.txt", "K182513.txt", "K181324.txt", "K173887.txt"
]:
    
    for num_tokens, lr in [(131702, 0.00)]:

        configs.append(

            TrainConfig(
                name=f"{file_name}_{DOC_NAME}_nt{num_tokens}",

                model=HFModelConfig(
                    pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct"
                ),

                dataset=CartridgeDatasetLatest.Config(
                    data_sources=[("/data/simran/dataset_K182513.pkl", None)], # Garbage
                    is_wandb=True,
                    label_type="logits",
                    top_k_logits=20,
                ),

                context=EvaporateContextConfig(
                    doc_id=DOC_NAME, 
                ),

                # Evals and generations
                generate_every_n_steps=1024,
                generate_datasets=[
                    GenerateDatasetConfig(
                        dataset= EvaporateMultipleChoiceGenerateDataset.Config(
                            doc_id=DOC_NAME, 
                            cot=False,
                        ),
                        name_for_wandb=f"evaporate_{DOC_NAME}_mc",
                    ),
                ],

                eval_every_n_steps=64,
                eval_datasets=[
                    EvalDatasetConfig(
                        name_for_wandb=f"evaporate_mc",
                        local_batch_size=16,
                        dataset=EvaporateEvalDataset.Config(
                            doc_id=DOC_NAME,
                            max_questions=256,
                            label_type="tokens",
                            data_sources=[]  # ignore this arg
                        )
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
                    project="cartridges",
                    tags=["evaporate", "icl", f"doc{DOC_NAME}"],
                    entity="hazy-research",
                ),
                output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
                global_batch_size=bs,
                local_batch_size=1,
            )
        )

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])


