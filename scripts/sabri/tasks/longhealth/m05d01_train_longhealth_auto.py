import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.configs.common_evals.finance_evals_anthropic import get_evals
from cartridges.initialization.strategies.first_n_tokens import KVCacheInitFromFirstNTokensOfContext
from cartridges.models.llama import LlamaForCausalLM
from cartridges.optim import CosWithWarmup
from cartridges.tasks.longhealth import LongHealthEvalDataset, LongHealthMultipleChoiceGenerateDataset
from cartridges.tasks.reglab import ReglabHousingQAGenerateDataset
from cartridges.train import EvalDatasetConfig, GenerateDatasetConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeDataset, CartridgeDatasetLatest
from cartridges.utils import WandBConfig

file_name = Path(__file__).stem

bs = 64


NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]


PROMPT_TYPES = [
    "simple_lhprompts",
    # "tree_lhprompts",
    # "legacy",
]

if NUM_PATIENTS == 10:
    data_sources = {
        "simple_lhprompts": [
            # 4096 max context length, long_health prompts 
            ("/data/sabri/code/Cartridges/output/2025-05-03-12-43-00-m05d03_generate_longhealth_auto_healthprompt/b128659b-4b18-48bf-8b98-5a3f2615bcca/artifact/dataset.pkl", None),
            ("/data/sabri/code/Cartridges/output/2025-05-03-13-00-10-m05d03_generate_longhealth_auto_healthprompt/bfbf8857-f4ba-41f9-81da-7e5ea8d300c6/artifact/dataset.pkl", None),
            ("/data/sabri/code/Cartridges/output/2025-05-03-13-16-45-m05d03_generate_longhealth_auto_healthprompt/8d067664-f598-4d91-94c3-6c9af5e14ee9/artifact/dataset.pkl", None),
            ("/data/sabri/code/Cartridges/output/2025-05-03-14-52-54-m05d03_generate_longhealth_auto_healthprompt/2fce1115-854c-4cb6-a5f7-411c75bbe58e/artifact/dataset.pkl", None),


            # 16384 max context length, long_health prompts 
            ("/data/sabri/code/Cartridges/output/2025-05-03-15-31-59-m05d03_generate_longhealth_auto_healthprompt/8634d8f0-e8be-4565-99d1-5a8744e3444b/artifact/dataset.pkl", None),
            ("/data/sabri/code/Cartridges/output/2025-05-03-16-18-33-m05d03_generate_longhealth_auto_healthprompt/d49d9081-51f6-4408-a486-2a8b2e443a36/artifact/dataset.pkl", None),
        ],
        "tree_lhprompts": [
            ("/data/sabri/code/Cartridges/output/2025-05-04-16-48-52-m05d03_generate_longhealth_auto_tree/dfadf61a-199b-4145-9ed5-5421bf590084/artifact/dataset.pkl", None),
        ],
        "legacy": [
            ("/data/sabri/code/Cartridges/output/2025-04-22-15-10-19-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/58acef64-5991-4174-8f6c-25de7a817596/artifact/dataset.pkl", None),
            ("/data/sabri/code/Cartridges/output/2025-04-22-19-28-13-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/75cec0ba-ab2b-4542-a114-99fb679b44eb/artifact/dataset.pkl", None),
            ("/data/sabri/code/Cartridges/output/2025-04-22-20-40-40-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/cc2c97e1-eb6a-467a-b86b-b541c148fed0/artifact/dataset.pkl", None),
            ("/data/sabri/code/Cartridges/output/2025-04-22-21-32-05-m04d22_generate_longhealth_generated_reformat_w_toc_npatients/559b979b-c2b4-44dd-98b8-6be995510561/artifact/dataset.pkl", None),
        ]
    }
elif NUM_PATIENTS == 20:
    data_sources = [
    ]
else:
    raise ValueError(f"Invalid number of patients: {NUM_PATIENTS}")




configs = [
    TrainConfig(
        name=FormatStringVariable(f"{file_name}_{patients_str}_{'+'.join(PROMPT_TYPES)}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
        model=HFModelConfig(
            pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
            model_cls=LlamaForCausalLM,
            attn_implementation="einsum",
        ),
        
        lr=lr,

        dataset=CartridgeDatasetLatest.Config(
            data_sources=sum([data_sources[p] for p in PROMPT_TYPES], start=[]),
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
                dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                    patient_ids=patient_ids, 
                    cot=True,
                ),
                name_for_wandb=f"longhealth_mc",
                num_samples=4,
                batch_size=16,
                temperature=0.3
            ),
        ],
        eval_every_n_steps=256,
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
            )
        ],
        kv_cache_initializer=KVCacheInitFromFirstNTokensOfContext.Config(max_tokens=num_tokens),
        loss_type="logits",
        epochs=16,

        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "longhealth", f"patients{patients_str}"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        distributed_backend="gloo" if socket.gethostname().startswith("mk-ix-") else "nccl",
        global_batch_size=bs,
        local_batch_size=1,
    )
    for lr in [
        # 4e-2,
        2e-2,
        # , 1e-2, 9e-3, 8e-3
    ]
    # for lr in [8e-3]
    for num_tokens in [
        # 2048, 
        512,
        # 6144, 
        # 2048, 
        # 2048, 
        # 1024
    ]
]

if __name__ == "__main__":
    pydrantic.main(configs)
