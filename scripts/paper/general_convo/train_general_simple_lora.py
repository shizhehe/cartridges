import os
from pathlib import Path
import pydrantic

from cartridges.configs.common_evals.finance_evals import FinanceEvals
from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from cartridges.tasks.finance import ( FinanceBenchContextConfig, FinanceBenchEvalDataset, FinanceBenchGenerateDataset, FinanceBenchMemorizationDataset )
from cartridges.tasks.mmlu import MMLUEvalDataset
from cartridges.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from cartridges.models.config import HFModelConfig, PeftConfig
from cartridges.datasets import ( CartridgeDatasetLatest )
from cartridges.utils import WandBConfig


file_name = Path(__file__).stem


configs = []
DOC_NAME = "AMD_2022_10K"
bs = 64


if DOC_NAME == "AMD_2022_10K":

    # Experiment: Cartridge V1 SimplePromptSampler
    dataset_sources = [
        ("/data/simran/Cartridges/outputs/2025-05-02-20-21-39-m05d01_generate_memorize_subcontext_v0/89925dfa-f385-41d6-adf0-ddcab78a22e2/artifact/dataset.pkl", None,),
        # ("/data/simran/Cartridges/outputs/2025-05-04-13-21-28-m05d04_auto_cartridge/bc9505b2-01d9-41c6-a04f-a15522f7ff00/artifact/dataset.pkl", None,),
        # ("/data/simran/Cartridges/outputs/2025-05-04-14-39-29-m05d04_auto_cartridge/9dfc953f-c8a7-49a4-adec-5ceee3e93885/artifact/dataset.pkl", None,),
        ("/data/simran/Cartridges/outputs/2025-05-04-15-33-06-m05d04_generate_reformatter_v0/c4fd690a-df33-409e-ac08-68fd3f4c15f3/artifact/dataset.pkl", None,),

        # two seed prompts
        ("/data/simran/Cartridges/outputs/2025-05-04-20-25-26-m05d04_auto_cartridge/cd9c9317-f1f6-4c48-b1ce-ae671f0109e0/artifact/dataset.pkl", None,),

        # three seed prompts
        # ("/data/simran/Cartridges/outputs/2025-05-04-20-35-35-m05d04_auto_cartridge/2fef1efd-aad1-4ff6-9566-710679c89565/artifact/dataset.pkl", None),
    ]
    dataset_mix = "seed2_sampler_65k_yesReformat"
else:
    raise ValueError(f"Unknown dataset: {DOC_NAME}")


configs = []

generate_evals, ppl_evals = get_evals(
    FinanceEvals,
    DOC_NAME,
    num_samples=16,  # RE: it's silly we have to specify this
    version_tag="v1",
    batch_size=16,
)

lora_rank = 384 // 2

lr = 5e-2
for num_tokens, lr in [(4096, 0.03)]:
    configs.append(

        TrainConfig(
            name=f"{file_name}_{DOC_NAME}_nt{num_tokens}_auto_data{dataset_mix}",
            model=HFModelConfig(
                pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
                tuning_method="peft",  # Use PEFT instead of custom prefix tuning
                peft=PeftConfig(
                    enabled=True,  # Enable PEFT
                    method="lora",  # Use LoRA
                    r=lora_rank,  # LoRA rank
                    alpha=int(2 * lora_rank),  # LoRA scaling factor (typically 2*rank)
                    dropout=0.05,  # LoRA dropout
                    # Updated target modules for LLaMA 3 architecture
                    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
            ),

            dataset=CartridgeDatasetLatest.Config(
                data_sources=dataset_sources,
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),

            generate_every_n_steps=1024,
            generate_datasets=[
                GenerateDatasetConfig(
                    name_for_wandb="finance-ppl-gt",
                    # local_batch_size=16,
                    dataset=FinanceBenchGenerateDataset.Config(
                        doc_names=[DOC_NAME],
                    ),
                ),
                *generate_evals,
            ],

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
                EvalDatasetConfig(
                    name_for_wandb="mmlu",
                    local_batch_size=16,
                    dataset=MMLUEvalDataset.Config(num_samples=512),
                ),
                
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
                project="cartridges",
                tags=["cache_tuning", "development"],
                entity="hazy-research",
            ),
            output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
            global_batch_size=bs,
            local_batch_size=4,
        )
    )

if __name__ == "__main__":
    for config in configs:
        pydrantic.main([config])

