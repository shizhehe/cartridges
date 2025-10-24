import os
from pathlib import Path
import socket

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.data.ruler.evals import NIAHGenerateDataset
from cartridges.initialization.text import KVFromText
from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.utils.wandb import WandBConfig
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset

file_name = Path(__file__).stem
bs = 4

NUM_PATIENTS = 10
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))

NUM_KEYS = (1, 1)
num_keys_str = f"k{NUM_KEYS[0]}_{NUM_KEYS[1]}"

MODEL = os.environ.get("MODEL", "llama")
if MODEL == "llama":
    DATA_SOURCES = {
        (1, 1): [
            "/Users/sabrieyuboglu/code/cartridges-internal/output/2025-09-23-19-08-24-m09d24_niah_synthesize/m09d24_niah_synthesize_llama-3.2-3b_n65536_k1_1-0/artifact/dataset.parquet",
            "/Users/sabrieyuboglu/code/cartridges-internal/output/2025-09-23-19-22-00-m09d24_niah_synthesize/m09d24_niah_synthesize_llama-3.2-3b_n65536_k1_1-0/artifact/dataset.parquet"
        ],

    }
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

BASE_PATH = os.path.join(
    os.environ["CARTRIDGES_DIR"], "cartridges/data/ruler/_data"
)
NUM_KEYS_TO_PATH = {
    (1, 1): f"{BASE_PATH}/llama_3.2_3b_instruct-l128000-n1-k128-v1_1-essay-key_words-val_numbers-74198fb4.json"
}

data_sources = DATA_SOURCES[NUM_KEYS]
niah_path = NUM_KEYS_TO_PATH[NUM_KEYS]

config = TrainConfig(
    model=model,
    kv_cache_initializer=KVFromText.Config(
        max_tokens=NUM_TOKENS
    ),
    
    lr=2e-2,
    epochs=2,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=source, type="local") for source in data_sources],
        top_k_logits=20,
        packed_seq_length=8192,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_eval_every_n_steps=16,
    generate_evals=[
        GenerationEvalConfig(
            dataset=NIAHGenerateDataset.Config(
                niah_path=niah_path, thinking=False,
            ),
            name_for_wandb=f"niah_mc",
            num_samples=8,
            override_max_tokens=256,
            temperature=0.3,
            batch_size=16,
        ),
        
    ],
    
    distributed_backend="nccl",

    wandb=WandBConfig(
        project="cartridges",
        tags=["train", "niah", num_keys_str],
        entity="hazy-research",
    ),
    output_dir=os.environ["CARTRIDGES_OUTPUT_DIR"],
    name=FormatStringVariable(f"{file_name}_lr{{lr}}_toks{{kv_cache_initializer.max_tokens}}_{num_keys_str}"),
)


if __name__ == "__main__":
    pydrantic.main(config)