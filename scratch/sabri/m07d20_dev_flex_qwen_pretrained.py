import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoTokenizer

from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
from cartridges.models.qwen.configuration_qwen3 import Qwen3Config

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.random import KVFromRandomText, KVFromRandomVectors

device = "cuda"

chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6b")
model = FlexQwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6b").to(device)

input_ids = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt").to(device)
seq_ids = torch.full_like(input_ids, 0, dtype=torch.long, device=device)[0]
position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
past_key_values = None

print("Running forward pass...")
out = model(input_ids, seq_ids=seq_ids, position_ids=position_ids, past_key_values=past_key_values)
print("✅ Forward pass completed")
out = model(input_ids, seq_ids=seq_ids, position_ids=position_ids, past_key_values=past_key_values)
print("Again")


