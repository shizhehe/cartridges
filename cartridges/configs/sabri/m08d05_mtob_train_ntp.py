import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.mtob.evals import MTOBKalamangToEnglishGenerateDataset
from cartridges.initialization.random import KVFromRandomText, KVFromRandomVectors
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeTrainDataset
from cartridges.utils import WandBConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset

file_name = Path(__file__).stem
bs = 4

NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "4096"))

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "llama":
    data_sources = [
        "/data/sabri/cartridges/2025-08-05-15-18-14-m08d05_mtob_synthesize_ntp/m08d05_mtob_synthesize_ntp_llama-3.1-8b_n65536-0/artifact/dataset.pkl"
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

configs = []
for lr in [2e-2]:
    config = TrainConfig(
        model=model,
        kv_cache_initializer=KVFromRandomText.Config(
            max_tokens=NUM_TOKENS
        ),
        
        lr=lr,
        epochs=2,
        global_batch_size=32,

        dataset=CartridgeTrainDataset.Config(
            data_sources=[
                (source, None)
                for source in data_sources
            ],
            top_k_logits=20,
            packed_seq_length=2048,
            packing_mode="truncate",
        ),

        save_every_n_steps=512,
        generate_every_n_steps=128,
        generate_evals=[
            GenerationEvalConfig(
                name_for_wandb=f"mtob-ke-test",
                dataset=MTOBKalamangToEnglishGenerateDataset.Config(use_cot=False),
                batch_size=16,
                generate_max_new_tokens=128,
                num_samples=1,
                temperature=0,
            ),
        ],
        eval_every_n_steps=512,
        eval_datasets=[],
        distributed_backend="gloo",

        wandb=WandBConfig(
            project="cartridges",
            tags=["train", "mtob"],
            entity="hazy-research",
        ),
        output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
        name=FormatStringVariable(f"{file_name}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}"),
    )
    configs.append(config)

if __name__ == "__main__":
    pydrantic.main(configs)