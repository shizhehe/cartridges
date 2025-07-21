import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from cartridges.models.qwen.modeling_qwen3 import (
    FlexQwen3Model,
)
from cartridges.models.qwen.configuration_qwen3 import Qwen3Config

from transformers import Qwen3Model, Qwen3Config
from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.random import KVFromRandomText, KVFromRandomVectors

device = "cuda"

def small_qwen_config():
    """Create a small Qwen3 config for testing."""
    num_layers = 3
    return Qwen3Config(
        vocab_size=1000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=num_layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
        use_cache=False,
        layer_types=["full_attention"] * num_layers,
    )


cartridge_len = 1024
seq_lens = [32, 64, 128, 64]
batch_size = len(seq_lens)
max_seq_len = max(seq_lens)
total_seq_len = sum(seq_lens)

seq_ids = torch.cat(
    [torch.full((seq_len,), idx, dtype=torch.long, device=device) for idx, seq_len in enumerate(seq_lens)]
)
input_ids = torch.randint(0, 1000, (1, total_seq_len)).to(device)
position_ids = torch.cat(
    [torch.arange(seq_len, device=device) for seq_len in seq_lens]
).unsqueeze(0)

padded_input_ids = torch.full((batch_size, max_seq_len), 0, dtype=torch.long, device=device)
start_idx = 0   
for i, seq_len in enumerate(seq_lens):
    padded_input_ids[i, :seq_len] = input_ids[0, start_idx:start_idx+seq_len]
    start_idx += seq_len
padded_position_ids = torch.stack(
    [torch.arange(max_seq_len, device=device) for _ in range(batch_size)]
)

config = small_qwen_config()
model = FlexQwen3Model(config).to(device)
ref_model = Qwen3Model(config).to(device)
ref_model.load_state_dict(model.state_dict())

# Check equivalence with no cache
# --- begin no cache ---
out = model(input_ids, seq_ids=seq_ids, position_ids=position_ids).last_hidden_state
ref_out_padded = ref_model(padded_input_ids).last_hidden_state

ref_out = torch.cat(
    [ref_out_padded[batch_idx, :seq_len] for batch_idx, seq_len in enumerate(seq_lens)],
    dim=0
).unsqueeze(0)
torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)
print("✅ Forward check passed (no cache)")
# --- end no cache ---


# Check equivalence with cache
# --- begin with cache ---
cache = KVFromRandomVectors.Config(max_tokens=cartridge_len).instantiate().initialize_kv_cache(
    tokenizer=None,
    model=ref_model,
    attn_config=AttnConfig(
        n_layers=config.num_hidden_layers,
        n_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    ),
)
cache.to(device)

out = model(input_ids, seq_ids=seq_ids, position_ids=position_ids, use_cache=True, past_key_values=cache).last_hidden_state
out.sum().backward()
keys_grad = cache.trainable_keys[0].grad.clone()
values_grad = cache.trainable_values[0].grad.clone()

cache.zero_grad()
cache.clear()

ref_out_padded = ref_model(padded_input_ids, use_cache=True, past_key_values=cache).last_hidden_state
ref_out = torch.cat(
    [ref_out_padded[batch_idx, :seq_len] for batch_idx, seq_len in enumerate(seq_lens)],
    dim=0
).unsqueeze(0)
ref_out.sum().backward()
ref_keys_grad = cache.trainable_keys[0].grad.clone()
ref_values_grad = cache.trainable_values[0].grad.clone()

torch.testing.assert_close(out, ref_out, atol=1e-3, rtol=1e-3)
print("✅ Forward check passed (with cache)")
torch.testing.assert_close(keys_grad, ref_keys_grad, atol=1e-2, rtol=1e-2)
torch.testing.assert_close(values_grad, ref_values_grad, atol=1e-2, rtol=1e-2)
print("✅ Backward check passed (with cache)")

# --- end with cache ---








