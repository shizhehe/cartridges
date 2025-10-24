import os
from pathlib import Path
import pydrantic
from pydrantic.variables import FormatStringVariable

from capsules.configs.common_evals.finance_evals import FinanceEvals
from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import ( KVCacheInitFromFirstNTokensOfContext )
from capsules.tasks.finance import ( FinanceBenchContextConfig, FinanceBenchEvalDataset, FinanceBenchGenerateDataset, FinanceBenchMemorizationDataset )
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.longhealth.context import LongHealthStructuredContextConfig

from capsules.tasks.mmlu import MMLUEvalDataset, MMLUGenerateDataset
from capsules.train import EvalDatasetConfig, TrainConfig, GenerateDatasetConfig
from capsules.config import HFModelConfig, PeftConfig
from capsules.datasets import ( CapsuleDatasetLatest )
from capsules.utils import WandBConfig


file_name = Path(__file__).stem


configs = []
bs = 64


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]


NUM_TOKENS = os.environ.get("NUM_TOKENS", "2048")
NUM_TOKENS = list(map(int, NUM_TOKENS.split(",")))

if NUM_PATIENTS == 10:

    dataset_sources = [
        ("hazy-research/capsules/generate_longhealth_simple_p10_s5_n65536:v0", None),
        ("hazy-research/capsules/generate_longhealth_simple_p10_s5_n65536:v1", None),
    ]
    # dataset_mix = "seed2_sampler_65k_yesReformat"
else:
    raise ValueError(f"Invalid number of patients: {NUM_PATIENTS}")


configs = []

lora_rank = 384 // 2

lr = 5e-2
for num_tokens, lr in [(4096, 0.03)]:
    configs.append(

        TrainConfig(
            name=FormatStringVariable(f"{file_name}_{patients_str}_lr{{lr}}_lora{lora_rank}_toks{{kv_cache_initializer.max_tokens}}"),
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

            dataset=CapsuleDatasetLatest.Config(
                data_sources=dataset_sources,
                max_sequence_length=1024,
                is_wandb=True,
                label_type="logits",
                top_k_logits=20,
            ),

            generate_every_n_steps=1024,
            generate_datasets=[
                GenerateDatasetConfig(
                    dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                        patient_ids=patient_ids, 
                        cot=True,
                    ),
                    name_for_wandb=f"longhealth_mc",
                    num_samples=8,
                    num_samples_final=8,
                    batch_size=2,
                    temperature=0.3
                ),
                # GenerateDatasetConfig(
                #     dataset=MMLUGenerateDataset.Config(),
                #     name_for_wandb=f"longhealth_mc",
                #     num_samples=8,
                #     num_samples_final=8,
                #     batch_size=16,
                #     temperature=0.3
                # ),
            ],

            context=LongHealthStructuredContextConfig(patient_ids=patient_ids),

            eval_every_n_steps=64,
            eval_datasets=[
                EvalDatasetConfig(
                    name_for_wandb="longhealth_mc",
                    local_batch_size=16,
                    dataset=LongHealthEvalDataset.Config(
                        patient_ids=patient_ids,
                        max_questions=256,
                        label_type="tokens",
                        data_sources=[]  # ignore this arg
                    )
                ),
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
    for config in configs:
        pydrantic.main([config])

