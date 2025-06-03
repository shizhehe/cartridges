import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

# from capsules.configs.common_evals.finance_evals_anthropic import get_evals
from capsules.kv_initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from capsules.models.llama import LlamaForCausalLM
from capsules.optim import CosWithWarmup
from capsules.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from capsules.tasks.reglab import ReglabHousingQAGenerateDataset, ReglabHousingCategoriesQAGenerateDataset
from capsules.tasks.reglab.utils import ALL_STATES
from capsules.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from capsules.config import HFModelConfig
from capsules.datasets import CapsuleDataset, CapsuleDatasetLatest
from capsules.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


NUM_STATES = 5
states_str = f"{NUM_STATES}states"
states = ALL_STATES[:NUM_STATES]

if "SLICES" in os.environ:
    SLICES = os.environ["SLICES"].split(",")
else:
    SLICES = [
        # "structuring",
        # "summarization",
        "aggregation",
        # "question",
        # "use_case",
    ]

data_sources = {
    "structuring": [
        "hazy-research/capsules/m05d05_generate_reglab_auto_5states_structuring_n8192_cot0.2:v1",
        # "/home/emilyryliu/capsules-outputs/2025-05-05-21-43-07-m05d05_generate_reglab_auto/m05d05_generate_reglab_auto_5states_structuring_n8192_cot0.2-0/artifact/dataset.pkl",
        # "hazy-research/capsules/m05d05_generate_reglab_auto_5states_structuring_n8192_cot0.2:v0",
    ],
    "summarization": [
        "hazy-research/capsules/m05d05_generate_reglab_auto_5states_summarization_n8192_cot0.2:v0",
    ],
    "aggregation": [
        "hazy-research/capsules/m05d05_generate_reglab_auto_5states_aggregation_n8192_cot0.2:v0",
    ],
    "question": [
        "hazy-research/capsules/m05d05_generate_reglab_auto_5states_question_n8192_cot0.2:v1",
    ],
    "use_case": [
        "hazy-research/capsules/m05d05_generate_reglab_auto_5states_use_case_n8192_cot0.2:v0",
    ]
}



configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_{states_str}_{'+'.join(SLICES)}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            model_cls=LlamaForCausalLM,
            attn_implementation="einsum",
        ),
        
        lr=lr,

        dataset=CapsuleDatasetLatest.Config(
            data_sources=[
                (source, None)
                for slc in SLICES
                for source in data_sources[slc]
            ],
            max_sequence_length=1024,
            is_wandb=True,
            label_type="logits",
            top_k_logits=20,
        ),
        
        save_every_n_steps=512,
        generate_every_n_steps=512,
        generate_max_new_tokens=512,
        generate_datasets=[
            GenerateDatasetConfig(
                dataset=ReglabHousingCategoriesQAGenerateDataset.Config(
                    states =states,
                    cot=True,
                ),
                name_for_wandb=f"reglab_categories_generate",
                num_samples=4,
                temperature=0.3,
            ),
        ],
        eval_every_n_steps=256,
        eval_datasets=[
            # EvalDatasetConfig(
            #     name_for_wandb="longhealth_mc",
            #     local_batch_size=16,
            #     dataset=LongHealthEvalDataset.Config(
            #         patient_ids=patient_ids,
            #         max_questions=256,
            #         label_type="tokens",
            #         data_sources=[]  # ignore this arg
            #     )
            # )
        ],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(max_tokens=num_tokens),
        loss_type="logits",
        epochs=2,

        wandb=WandBConfig(
            project="capsules",
            tags=["train", "longhealth", f"{states_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CAPSULES_OUTPUT_DIR"],
        distributed_backend="gloo" if socket.gethostname().startswith("mk-ix-") else "nccl",
        global_batch_size=bs,
        local_batch_size=4,
    )
    for lr in [
        # 4e-2,
        2e-2,
        # , 1e-2, 9e-3, 8e-3
    ]
    # for lr in [8e-3]
    for num_tokens in [
        2048, 
        # 512,
        # 6144, 
        # 2048, 
        # 2048, 
        # 1024
    ]
]

if __name__ == "__main__":
    pydrantic.main(configs)
