import random
import pytest
import os
from pathlib import Path
import socket

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.models.qwen.configuration_qwen3 import Qwen3Config

from transformers import Qwen3Model, Qwen3Config
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.random import KVFromRandomText, KVFromRandomVectors


import pydrantic
from pydrantic.variables import FormatStringVariable
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cartridges.initialization.random import KVFromRandomText
from cartridges.train import PerplexityEvalConfig, GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import CartridgeTrainDataset
from cartridges.utils import WandBConfig

device = "cuda"



cartridge_len = 1024



model = FlexQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-4b").to(device).to(torch.bfloat16)   


# Check equivalence with cache
# --- begin with cache ---
cache = KVFromRandomVectors.Config(max_tokens=cartridge_len).instantiate().initialize_kv_cache(
    tokenizer=None,
    model=model,
    attn_config=AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    ),
)
cache.to(device).to(torch.bfloat16)

optimizer = torch.optim.Adam(cache.parameters(), lr=1e-2)

data_sources = [
    "/home/sabri/cartridges/outputs/2025-07-13-09-04-32-m07d11_longhealth_synthesize/m07d11_longhealth_synthesize_p10_n65536-0/artifact/dataset.pkl"
]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4b")

config =CartridgeTrainDataset.Config(
    data_sources=[
        (source, None)
        for source in data_sources
    ],
    max_sequence_length=1024,
    is_wandb=True,
    label_type="logits",
    top_k_logits=20,
)
dataset = config.instantiate(tokenizer=tokenizer)


dataloader = DataLoader(
    dataset,
    collate_fn=dataset.collate,
    batch_size=1
)
local_rank = "cuda"

for i, batch in enumerate(dataloader):
    print(f"iteration {i}")
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

        out = model(
            input_ids=batch.input_ids.to(device), 
            seq_ids=batch.element_ids.to(device), 
            position_ids=batch.position_ids.to(device), 
            use_cache=True, 
            past_key_values=cache
        )
        topk_pred_logprobs = F.log_softmax(out.logits, dim=-1)[
            0, 
            batch.topk_token_idxs.to(local_rank), 
            batch.topk_token_ids.to(local_rank)
        ] 

        # ce is sum -p(x)logq(x), where p is the true distr and q is the model distr
        ce_by_token = (
            -batch.topk_logprobs.to(local_rank).exp()  # p(x), true distr
            * topk_pred_logprobs  # q(x), model distr
        ).sum(dim=-1)

        loss = (ce_by_token.mean())
    # loss.backward()
    # optimizer.stkep()
    # optimizer.zero_grad()
    cache.clear()









# --- end with cache ---








